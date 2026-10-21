"""Frozen, train-history title-BM25 candidate retrieval for Amazon.

No label input in retrieval. All candidates except appended target are
unseen items. This is a researcher-defined hard-negative baseline,
NOT an official LLMRank Amazon retrieval file.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import log
import random
import re

STOPWORDS = {
    "the","a","and","for","with","to","of","in","on","by","from",
    "pack","set","new","item","product","size","one","an","is",
}

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+",str(text).lower())
            if len(w)>2 and w not in STOPWORDS]

class TitleBM25:
    def __init__(self,titles):
        self.n=len(titles)
        self.postings=defaultdict(list)
        self.doc_len={}
        for item,title in titles.items():
            counts=Counter(tokenize(title))
            length=sum(counts.values())
            self.doc_len[item]=length
            for token,frequency in counts.items():
                self.postings[token].append((item,frequency))
        self.avg_len=sum(self.doc_len.values())/max(self.n,1)
    def select(self,history,blocked,titles,seed,k=19):
        if k<=0:
            raise ValueError("candidate budget must be positive")
        query=Counter()
        for item in history[-20:]:
            query.update(set(tokenize(titles[item])) if item in titles else ())
        # Prefer discriminative terms, avoiding very frequent product tokens.
        rare=sorted(query,key=lambda term:(
            -log(1+(self.n-len(self.postings.get(term,()))+0.5)
                 /(len(self.postings.get(term,()))+0.5)),
            term))
        scores=defaultdict(float)
        for term in rare[:16]:
            posts=self.postings.get(term,())
            df=len(posts)
            if not df:
                continue
            idf=log(1+(self.n-df+0.5)/(df+0.5))
            for item,tf in posts:
                if item in blocked:
                    continue
                norm=1.2*(0.25+0.75*self.doc_len[item]/max(self.avg_len,0.1))
                scores[item]+=idf * (tf*2.2)/(tf+norm)
        ranked=sorted(scores,key=lambda i:(-scores[i],i))[:k]
        # Deterministic fallback; same candidate size even when titles sparse.
        if len(ranked)<k:
            rng=random.Random(seed)
            eligible=[i for i in sorted(titles) if i not in blocked and i not in ranked]
            if len(eligible)<k-len(ranked):
                raise ValueError("catalog too small for frozen candidate protocol")
            ranked+=rng.sample(eligible,k-len(ranked))
        return ranked
