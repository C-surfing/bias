"""Reproducible Amazon 2014 5-core -> sequential RSI dataset conversion.

Sources: McAuley/SNAP reviews_<Category>_5.json.gz and meta_<Category>.json.gz.
Do not use eval() on untrusted metadata: only ast.literal_eval for the
nonstandard Python-dict serialization of the original dataset.
"""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

NAMES = {"Beauty":"Beauty","Sports":"Sports_and_Outdoors","CDs":"CDs_and_Vinyl"}

def decode(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)

def normalize(s):
    return " ".join(str(s).split())[:180]

def prepare(category, source, destination, min_rating=0):
    src = Path(source)
    dst = Path(destination)
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True)
    origin = NAMES[category]
    reviews = src / f"reviews_{origin}_5.json.gz"
    metadata = src / f"meta_{origin}.json.gz"
    if not reviews.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Missing Amazon 2014 review/meta pair: {reviews}, {metadata}")
    user_events = defaultdict(dict)
    used_items = set()
    num_reviews = 0
    with gzip.open(reviews, "rt", encoding="utf-8") as f:
        for line in f:
            row = decode(line)
            num_reviews += 1
            if float(row.get("overall", 0)) < min_rating:
                continue
            user, asin = row.get("reviewerID"), row.get("asin")
            ts = row.get("unixReviewTime")
            if not user or not asin or ts is None:
                continue
            ts = int(ts)
            previous = user_events[user].get(asin)
            if previous is None or ts > previous:
                user_events[user][asin] = ts
            used_items.add(asin)
    titles = {}
    with gzip.open(metadata, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                row = decode(line)
            except (SyntaxError, ValueError):
                continue
            asin = row.get("asin")
            if asin in used_items:
                title = normalize(row.get("title") or "")
                if title and title.lower() not in {"nan","none"}:
                    titles[asin] = title
    # Metadata missingness is explicitly counted; never invent placeholder titles.
    count = 0
    output_lines = []
    for user in sorted(user_events):
        items = sorted(((ts, asin) for asin,ts in user_events[user].items() if asin in titles),
                       key=lambda v:(v[0],v[1]))
        if len(items) < 5:
            continue
        for ts, asin in items:
            output_lines.append(f"{user}\t{asin}\t{ts}\n")
        count += 1
    included = {line.split("\t")[1] for line in output_lines}
    with (dst / f"{category}.item").open("w", encoding="utf-8") as f:
        f.write("item_id:token\ttitle:token_seq\n")
        for asin in sorted(included):
            f.write(f"{asin}\t{titles[asin]}\n")
    with zipfile.ZipFile(dst / f"{category}.inter.zip","w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        with z.open(f"{category}.inter","w") as stream:
            stream.write(b"user_id:token\titem_id:token\ttimestamp:float\n")
            for line in output_lines:
                stream.write(line.encode("utf-8"))
    def sha256(path):
        with path.open("rb") as f:
            return hashlib.file_digest(f,"sha256").hexdigest()
    provenance = {
        "dataset":category, "source_release":"McAuley Amazon Product Data 2014, 5-core",
        "reference":"https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html",
        "source_reviews":reviews.name,"source_metadata":metadata.name,
        "source_sha256":{
            "reviews":sha256(reviews),"metadata":sha256(metadata)
        },
        "min_rating":min_rating,
        "reviews_before_filter":num_reviews, "users_retained":count,
        "items_retained":len(included),"interactions_retained":len(output_lines),
        "missing_title_items":len(used_items - titles.keys()),
        "split":"chronological leave-two-out performed by experiment runner",
        "metadata_rule":"Titles must be present; deduplicate user-item by most recent timestamp",
    }
    (dst/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    return provenance

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=NAMES.keys(),required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--min-rating",type=float,default=0)
    a=p.parse_args()
    print(json.dumps(prepare(a.dataset,a.source,a.output,a.min_rating),indent=2))

if __name__=="__main__":
    main()
