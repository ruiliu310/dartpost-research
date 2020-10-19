#!/usr/bin/env python
# coding: utf-8


from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import beta

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import argparse
import pickle


sns.set_theme("paper", "white")


df_users = pd.read_csv("./DSock/Active_Users.csv", index_col=0)
print(df_users.shape)


df_evals = pd.read_csv("./DSock/human_eval_analysis/attitude_tracking.csv")

df_evals = pd.merge(df_users, df_evals, how="left", left_on="MTurk_ID", right_on="id").rename({"id_x": "id"}, axis=1)

df_evals.index = [f"u{u}" for u in df_evals["id"]]


topics = list("abcdefgh")
topics_num = list(range(9))
deltas = ["10", "21", "32", "43", "54"]


timeline = pd.date_range("2020-09-03-03:59:59", periods=6, tz="utc")


def get_change(u, t): return df_evals.loc[u][[f"{t}_delta_{d}" for d in deltas]].tolist()


u_change = {
    u: {t: (np.array(get_change(u, t)) != 0).sum() for t in topics}
    for u in df_evals.index
}


# ## get events
#
# + post events -> nodes
# + comment events -> nodes
# + view events -> nodes
# + like events -> nodes
# + influence -> edge


df_posts = pd.read_csv("./DSock/direct_influence/posts_with_senti_and_topic.csv", parse_dates=["createdAt"])
df_comts = pd.read_csv("./DSock/direct_influence/comments_with_senti_and_topic.csv", delimiter=",",
                       parse_dates=["createdAt"]).dropna(subset=["CommenterId", "PostId"])
df_views = pd.read_csv("./DSock/postViews.csv", delimiter="|",
                       parse_dates=["createdAt"]).dropna(subset=["UserId", "PostId"])
df_likes = pd.read_csv("./DSock/Likes.csv", delimiter="|",
                       parse_dates=["createdAt"]).dropna(subset=["UserId", "PostId"])
df_repts = pd.read_csv("./DSock/ReportUsers.csv", delimiter="|", parse_dates=["createdAt"])

print(f"posts: {df_posts.shape}, comments: {df_comts.shape}, views: {df_views.shape}, likes: {df_likes.shape}, reports: {df_repts.shape}")


df_likes["UserId"] = df_likes["UserId"].astype(int)
df_likes["PostId"] = df_likes["PostId"].astype(int)

df_comts["CommenterId"] = df_comts["CommenterId"].astype(int)
df_comts["PostId"] = df_comts["PostId"].astype(int)


df_annot_posts = pd.read_csv("./DSock/posts_anno_labels.csv")
df_posts = df_posts.merge(df_annot_posts[["post_id", "majority_topic_label",
                                          "majority_sent_label"]], left_on="PostID", right_on="post_id")


df_annot_comts = pd.read_csv("./DSock/comments_anno_labels.csv")
df_comts = df_comts.merge(df_annot_comts[["comment_id", "majority_topic_label",
                                          "majority_sent_label"]], left_on="id", right_on="comment_id")


# ### create influence grap

post_comments = {f"p{p}": [] for p in df_posts["PostID"].values}
for c, p in df_comts[["id", "PostId"]].values:
    if f"p{p}" in post_comments:
        post_comments[f"p{p}"].append(f"c{c}")

G = nx.DiGraph()
for u, uname, obsr, sock in df_users[["id", "username", "isObserver", "isPuppet"]].values:
    G.add_node(f"u{u}", id=u, kind="user", name=uname, observer=(obsr == "t"), sock=(sock == "t"))
    if obsr == "t":
        G.nodes[f"u{u}"]["color"] = "obsr"
    elif sock == "t":
        G.nodes[f"u{u}"]["color"] = "sock"
    else:
        G.nodes[f"u{u}"]["color"] = "part"

for p, u, t, polar, score, topic in df_posts[["PostID", "AuthorId", "createdAt", "majority_sent_label", "scores", "majority_topic_label"]].values:
    if f"u{u}" in G.nodes:
        G.add_node(f"p{p}", id=p, kind="post", time=t, user=u, polar=polar, score=score, topic=topic)

for c, u, p, t, polar, score, topic in df_comts[["id", "CommenterId", "PostId", "createdAt", "majority_sent_label", "scores", "majority_topic_label"]].values:
    if f"u{u}" in G.nodes and f"p{p}" in G.nodes:
        G.add_node(f"c{c}", id=c, kind="comt", time=t, user=u, polar=polar, score=score, topic=topic)
        G.add_edge(f"c{c}", f"p{p}", kind="known", time=t, weight=1)
        # G.add_edge(f"p{p}", f"c{c}", kind="known", time=t, weight=1)

for l, u, p, t in df_likes[["id", "UserId", "PostId", "createdAt"]].values:
    if f"u{u}" in G.nodes and f"p{p}" in G.nodes:
        G.add_node(f"l{l}", user=f"u{u}", post=f"p{p}", time=t, id=l, kind="like",
                   topic=G.nodes[f"p{p}"]["topic"], polar=G.nodes[f"p{p}"]["polar"], score=1)
        G.add_edge(f"u{u}", f"l{l}", time=t, weight=1, kind="infer")
        G.add_edge(f"l{l}", f"p{p}", time=t, weight=1, kind="known")

for u, p, t, v, s in df_views[["UserId", "PostId", "createdAt", "id", "singleView"]].values:
    if f"u{u}" in G.nodes and f"p{p}" in G.nodes:
        G.add_node(f"v{v}", user=f"u{u}", post=f"p{p}", time=t, id=v, kind="view",
                   topic=G.nodes[f"p{p}"]["topic"], polar=G.nodes[f"p{p}"]["polar"], score=1)
        G.add_edge(f"u{u}", f"v{v}", time=t, weight=1, kind="infer")
        G.add_edge(f"v{v}", f"p{p}", time=t, weight=1, kind="infer")
        if s == "t":
            for c in post_comments[f"p{p}"]:
                if c in G.nodes:
                    G.add_edge(f"v{v}", c, time=t, weight=1, kind="infer")


user_view = {u: set() for u in G if u[0] == "u"}
for u, p, t, v, s in df_views[["UserId", "PostId", "createdAt", "id", "singleView"]].values:
    if f"u{u}" in G.nodes and f"p{p}" in G.nodes and s != "t":
        user_view[f"u{u}"].add(f"v{v}")

single_view = {u: set() for u in G if u[0] == "u"}
for u, p, t, v, s in df_views[["UserId", "PostId", "createdAt", "id", "singleView"]].values:
    if f"u{u}" in G.nodes and f"p{p}" in G.nodes and s == "t":
        single_view[f"u{u}"].add(f"v{v}")

print(f"user view: {sum([len(user_view[u]) for u in user_view])}")
print(f"single view: {sum([len(single_view[u]) for u in single_view])}")

for p in tqdm(G):
    if p[0] in ["p", "c"]:
        u = f"u{G.nodes[p]['user']}"
        for v in user_view[u] | single_view[u]:
            if G.nodes[v]["time"] <= G.nodes[p]["time"]:
                G.add_edge(p, v, time=G.nodes[v]["time"], weight=1, kind="infer")


# ## On average, one user will read the same post almost 10 times


up = [(G.nodes[n]["user"], G.nodes[n]["post"]) for n in G.nodes if n[0] == "v"]

print(f"{len(up)} {len(set(up))} {len(up)/len(set(up))}")


print(f"{len(G)} {len(G.edges)}")


# #### define colors and styles for graph drawin


use_color = sns.color_palette("tab10")


line_map = {
    "view": "-",
    "like": ".",
}

color_map = {
    "view": 4,
    "like": 1,

    "known": 4,
    "infer": 4,

    "part": 2,
    "sock": 3,
    "obsr": 7,

    "make": 5,
    "comt": 9,
    "post": 0,

    "user": 8,
}

style_map = {
    "post": "o",
    "comt": "s",
}


def describe_graph(graph):
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "components": nx.number_weakly_connected_components(graph),
        "density": nx.density(graph),
        "diameter": nx.diameter(graph.to_undirected()) if nx.is_connected(graph.to_undirected()) else None,
        "degree": len(graph.edges)/len(graph.nodes),
    }


def draw_graph(subG, prog=None, edge_label=False, node_label=False, node_size=None, figsize=(10, 10), pr_value=None):
    if prog in ["dot", "neato"]:
        pos = nx.drawing.nx_agraph.graphviz_layout(subG, prog=prog)
    else:
        pos = nx.spring_layout(subG, seed=5, iterations=10)

    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()

    nx.draw(
        subG, ax=ax,
        pos=pos,
        with_labels=node_label,
        node_color=[use_color[color_map[subG.nodes[n]["kind"]]]
                    if n[0] != "u" else use_color[color_map[subG.nodes[n]["color"]]]
                    for n in subG.nodes],
        edge_color=[use_color[color_map[subG.edges[e]["kind"]]] for e in subG.edges],
        labels={n: f"{n}\n{subG.nodes[n]['polar']}" if subG.nodes[n]['kind'] != "user" else n for n in subG.nodes},
        font_size=10,
        node_size=[pr_value[n]*len(subG)*100 for n in subG] if pr_value else 300,
    )

    nx.draw_networkx_edge_labels(
        subG, pos=pos, ax=ax,
        edge_labels={
            e: f"{subG.edges[e]['time']:%d-%H:%M}" for e in subG.edges} if edge_label else {e: "" for e in subG.edges}
    )
    print(describe_graph(subG))


node_name = "u2337"
topic = 7

node_list = [e[1] for e in G.out_edges(node_name) if G.nodes[e[1]]["topic"] == topic][:10] + [node_name]
subG = nx.subgraph(G, node_list)
# draw_graph(subG, edge_label=True, node_label=True, figsize=(6, 6))


# ## full influence grap


node_name = "u2337"
topic = 4
polarity = 0

descendants_full = [n for n in nx.descendants(G, node_name)
                    if G.nodes[n]["topic"] == topic and G.nodes[n]["polar"] == polarity]

np.random.seed(4)
# descend_full = np.random.choice(descend_full, 65, replace=False).tolist()

subG = G.subgraph(descendants_full + [node_name]).copy()
print(len(subG))

subG = subG.subgraph([n for n in subG if nx.has_path(subG, node_name, n)])
print(len(subG))

# draw_graph(subG, prog="neato", edge_label=False, node_label=False, pr_value=None)


subsubg = subG.copy()
print(f"nodes {len(subsubg)}, edges: {len(subsubg.edges)}")
subsubg.remove_nodes_from([n for n in subsubg if n[0] == "v"])
print(f"nodes {len(subsubg)}, edges: {len(subsubg.edges)}")

# draw_graph(subsubg, prog=None, node_label=False, edge_label=False)


print(describe_graph(subG))
print(describe_graph(subsubg))


attention_window = pd.Timedelta("1 day")
# moment_time = timeline[1]
# pr_alpha = 0.85
# beta_a = 0.5
# beta_b = 0.5

# node_name = "u2337"
# topic = 4
# polarity = 0

# print(f"{attention_window}, {moment_time}")


def compute_moment_pagerank(G, node_name, topic, polarity, pr_alpha, beta_a, beta_b, moment_time, attention_window, verbose=False):
    moment_nodes = [n for n in nx.descendants(G, node_name)
                    if G.nodes[n]["topic"] == topic and G.nodes[n]["polar"] == polarity] + [node_name]

    moment_graph = G.subgraph(moment_nodes).copy()
    if verbose:
        print(f"nodes {len(moment_graph)}, edges: {len(moment_graph.edges)}")
        print(f"comments {len([n for n in moment_graph if n[0] == 'c'])}")
    moment_graph.nodes[node_name]["time"] = moment_time

    # remove edges outside attention window
    remove_edges = [e for e in moment_graph.edges if moment_graph.edges[e]["time"] > moment_time]
    for n in moment_graph:
        if n[0] in ["p", "u"]:
            t = moment_graph.nodes[n]["time"]
            remove_edges += [e for e in moment_graph.out_edges(n) if e[1][0] ==
                             "v" and not t - attention_window < moment_graph.edges[e]["time"] <= t]

    if verbose:
        print(f"remove edges {len(remove_edges)}")
        print(f"comments {len([n for n in moment_graph if n[0] == 'c'])}")
    moment_graph.remove_edges_from(remove_edges)
    if verbose:
        print(f"nodes {len(moment_graph)}, edges: {len(moment_graph.edges)} <- attention window")
        print(f"comments {len([n for n in moment_graph if n[0] == 'c'])}")

    # for source node node_name only
    moment_graph = moment_graph.subgraph([n for n in moment_graph if nx.has_path(moment_graph, node_name, n)]).copy()
    if verbose:
        print(f"nodes {len(moment_graph)}, edges: {len(moment_graph.edges)} <- path from observer")
        print(f"comments {len([n for n in moment_graph if n[0] == 'c'])}")

    if verbose:
        print(f"nodes {len(moment_graph)}, edges: {len(moment_graph.edges)} <- remove isolates")

    beta_rv = beta(a=0.5, b=0.5)

    for node in moment_graph:
        elist = sorted(moment_graph.out_edges(node), key=lambda e: G.edges[e]["time"])
        x = np.linspace(0, 1, len(elist)+2)[1:-1]
        y = beta_rv.pdf(x)
        y = y / y.sum()
        for e, w in zip(elist, y):
            moment_graph.edges[e]["weight"] = w

    pr_value = nx.pagerank_numpy(moment_graph, alpha=pr_alpha, weight="weight")

    return moment_graph, pr_value


# moment_graph, pr_value = compute_moment_pagerank(
#     G, node_name, topic, polarity, pr_alpha, beta_a, beta_b, moment_time, attention_window, verbose=True)


# draw_graph(moment_graph, prog="neato", node_label=False, edge_label=False, pr_value=pr_value)


obsr_list = [f"u{u}" for u in df_users[df_users["isObserver"] == "t"]["id"]]


# from multiprocessing import Pool

def short_compute_pagerank(n, t, topic, polar, pr_alpha, beta_a, beta_b):
    g, pr = compute_moment_pagerank(
        G, n, topic, polar, pr_alpha, beta_a, beta_b, timeline[t], attention_window)
    subg = g.copy()
    subg.remove_nodes_from([n for n in subg if n[0] == "v"])
    return {"full_graph": describe_graph(g), "exps_graph": describe_graph(subg), "pr": pr}


def do_params(pr_alpha, beta_a, beta_b):
    keyq = [(node_name, t, topic, polar, pr_alpha, beta_a, beta_b)
            for node_name in obsr_list for t in range(1, 6) for topic in range(8) for polar in [0, 2]
            ]
    print(f"jobs: {len(keyq)}")

    # valueq = Parallel(n_jobs=30, backend="threading")(delayed(short_compute_pagerank)(*tup) for tup in tqdm(keyq))
    pool = Pool(30)
    valueq = pool.starmap(func=short_compute_pagerank, iterable=tqdm(keyq), chunksize=1)
    return dict(zip(keyq, valueq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", "-p", type=float, default=0.9)
    parser.add_argument("--ba", "-a", type=float, default=0.5)
    parser.add_argument("--bb", "-b", type=float, default=0.5)

    parser.add_argument("--gen", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--graph", type=lambda x: x.lower() == "true", default=False)
    args = parser.parse_args()

    print(args)

    if args.gen:
        cmds = [f"python compute_pr.py -p {p} -a {a} -b {b}"
                for p in [0.85, 0.9, 0.7, 0.5, 0.3, 0.1]
                for a in [0.5, 0.9, 0.7, 0.3, 0.1]
                for b in [0.5, 0.9, 0.7, 0.3, 0.1]
                ]
        script = "\n".join(cmds)
        with open("./pr_script.sh", "w") as fp:
            fp.write(script)
    else:
        d = do_params(pr_alpha=args.pr, beta_a=args.ba, beta_b=args.bb)

        path = Path(f"res/pagerank/{args.pr}-{args.ba}-{args.bb}.pkl")
        with open(path, "wb") as fp:
            pickle.dump(d, fp)
