import re
import pathlib
from ._abbr import *


def loc(f):
    return pathlib.Path(__file__).parent.absolute() / f


__w_3_plus = re.compile(r"\w{3,}")
__w_ = re.compile(r"\w+")

names = set(
    [x.strip() for x in (open(loc("names_decode.csv")).readlines()[:1_000_000])]
)
names = names | set(["ford"])
# names from https://github.com/philipperemy/name-dataset


def name_preprocessing(z, remove_people=True):
    z = z.lower().replace("'", "")
    z = z.replace("-redh", "").replace("-old", "").replace("-new", "")
    z = (
        z.split("-pre")[0]
        .split("-adr")[0]
        .split("division of")[-1]
        .split("known as")[-1]
        .split("-consolidated")[0]
    )
    z = re.sub(r"(?=\w+)our\b", r"or", z)
    z = re.sub(r"old$|new$", r"", z)
    z = re.sub(r"(?=\w+)tt\b", r"t", z)
    #  z = re.sub(r'(?=(\w+))([a-zA-Z])\2?',r'\2',z)
    z = re.sub(r"(?=\w+)er\b", r"ers", z)  # to not match e.g. glove vs glover
    z = z.replace("`", "").replace("& company", "").replace("& companies", "")
    z = re.sub(r"\bco\.? inc\b", r"inc", z)
    z = re.sub(r"\bco\.? ltd\b", r"inc", z)
    z = re.sub(r"\bthe\b", "", z)
    z = re.sub(r"\b[a-z]\.(?!\w)\b", "", z)
    z = re.sub(r"\bjr\.\b", "", z)
    z = re.sub(r"\bsr\.\b", "", z)
    z = " ".join(re.findall(r"[\w\d]+", z))
    # combining single words...
    a = "".join(re.findall(r"\b\w\s\b", z))
    if a:
        b = a.replace(" ", "")
        z = z.replace(a, b + " ")

    # TODO: refactor the code to a function
    for string, adj_string in [
        ("i", ""),
        ("ii", ""),
        ("iii", ""),  # ('iv',''),('v',''),
        #  ('vi',''),('vii',''),('viii',''),('ix',''),('x','')
    ]:
        if string.startswith("i"):
            continue
        if string.startswith("("):
            z = re.sub(string, " " + adj_string, z, flags=re.IGNORECASE)
        else:
            z = re.sub(
                r"(?<!\w)" + string + r"(?!\w)",
                " " + adj_string,
                z,
                flags=re.IGNORECASE,
            )
    z = abbr_adj(z)
    ws = __w_.findall(z)
    if remove_people:
        counter = 0
        if len(ws) > 1:
            for w in ws:
                if w in names and len(w) > 1:
                    counter += 1

        if len(set([b for a, b in abbr if b != ""]) & set(ws)) == 0:
            if counter >= 2 and counter / len(set(ws) - set(["matchit"])) >= 0.6:
                return
    return z.strip().lower()
