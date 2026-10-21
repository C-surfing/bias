"""Small expression DSL for LLM-proposed reranking scores; no Python exec/eval.

Variables: base (LLM rank vote score), pop (train popularity), match (history
title-token overlap). All normalized 0..1 and computed without heldout target.
"""
import ast
import math

VARS={"base","score","pop","match","recent_match","user_pop",
      "pop_excess","novelty","head","rank_norm","history_len","pop_percentile"}
FUNCS={"abs":abs,"min":min,"max":max,
       "sqrt":math.sqrt,"log1p":math.log1p,
       "clamp":lambda value,low,high:min(high,max(low,value))}
BIN={ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,
     ast.Mult:lambda a,b:a*b,ast.Div:lambda a,b:a/b}
UNARY={ast.USub:lambda a:-a,ast.UAdd:lambda a:+a}

def validate(expr):
    if not isinstance(expr,str) or not 1<=len(expr)<=320:
        raise ValueError("rerank formula must contain 1..160 characters")
    try:
        tree=ast.parse(expr,mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid score expression") from exc
    if sum(1 for _ in ast.walk(tree))>90:
        raise ValueError("score expression too complex")
    for node in ast.walk(tree):
        if isinstance(node,(ast.Expression,ast.Load)):
            continue
        if isinstance(node,ast.IfExp):
            continue
        if (isinstance(node,ast.Compare) and len(node.ops)==1 and
                type(node.ops[0]) in (ast.Gt,ast.GtE,ast.Lt,ast.LtE,ast.Eq,ast.NotEq)):
            continue
        if isinstance(node,(ast.Gt,ast.GtE,ast.Lt,ast.LtE,ast.Eq,ast.NotEq)):
            continue
        if isinstance(node,ast.BinOp) and type(node.op) in BIN:
            continue
        if isinstance(node,ast.UnaryOp) and type(node.op) in UNARY:
            continue
        if isinstance(node,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.UAdd,ast.USub)):
            continue
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCS and not node.keywords:
            continue
        if isinstance(node,ast.Name) and node.id in VARS|FUNCS.keys():
            continue
        if isinstance(node,ast.Constant) and type(node.value) in (int,float) and math.isfinite(node.value) and abs(node.value)<=10:
            continue
        raise ValueError(f"forbidden score expression feature: {type(node).__name__}")
    return tree

def score(tree,features):
    def visit(node):
        if isinstance(node,ast.Expression):
            return visit(node.body)
        if isinstance(node,ast.Constant):
            return float(node.value)
        if isinstance(node,ast.Name) and node.id in features:
            return float(features[node.id])
        if isinstance(node,ast.IfExp):
            return visit(node.body) if visit(node.test) else visit(node.orelse)
        if isinstance(node,ast.Compare) and len(node.ops)==1:
            a,b=visit(node.left),visit(node.comparators[0])
            op=node.ops[0]
            return float((a>b if isinstance(op,ast.Gt) else
                          a>=b if isinstance(op,ast.GtE) else
                          a<b if isinstance(op,ast.Lt) else
                          a<=b if isinstance(op,ast.LtE) else
                          a==b if isinstance(op,ast.Eq) else a!=b))
        if isinstance(node,ast.BinOp) and type(node.op) in BIN:
            return BIN[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node,ast.UnaryOp) and type(node.op) in UNARY:
            return UNARY[type(node.op)](visit(node.operand))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCS:
            return FUNCS[node.func.id](*(visit(a) for a in node.args))
        raise ValueError("unsupported score node")
    try:
        value=visit(tree)
    except (ArithmeticError,OverflowError) as exc:
        raise ValueError("score expression arithmetic error") from exc
    if not math.isfinite(value) or abs(value)>1e6:
        raise ValueError("rerank formula produced nonfinite or extreme value")
    return value
