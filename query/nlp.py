import spacy

nlp = spacy.load("en_core_web_sm")

boilerplate_verbs_score = -1.0
BOILERPLATE_VERBS = {
    "consider",
    "discuss",
    "review",
    "present",
    "introduce",
    "outline",
    "describe",
    "summarize",
    "organize",
    "revisit",
    "recall",
    "note",       
    "mention",
    "highlight",
    "illustrate",    
    "emphasize",
    "focus",
    "explore",
    "investigate",   
    "examine",
    "compare",
    "contrast",
    "follow",        
} 

structural_verbs_score = -1.5
STRUCTURAL_VERBS = {
    "begin",
    "end",
    "start",
    "continue",
    "proceed",
    "turn",
    "conclude",
    "return",
    "refer",
    "point",
}

content_verbs_score = 1.0
CONTENT_VERBS = {
    "define",
    "prove",
    "show",
    "demonstrate",
    "establish",
    "derive",
    "imply",
    "characterize",
    "classify",
    "construct",
    "compute",
    "estimate",
    "bound",
    "approximate",
    "optimize",
    "minimize",
    "maximize",
    "converge",
    "diverge",
    "satisfy",
    "violate",
    "preserve",
    "exhibit",
    "exist",
    "hold",
    "fail",
}

relation_verbs_score = 1.0
RELATION_VERBS = {
    "depend",
    "relate",
    "correspond",
    "associate",
    "map",
    "transform",
    "reduce",
    "extend",
    "generalize",
    "specialize",
    "embed",
    "approximate",
} 

reporting_verbs_score = 0.5
REPORTING_VERBS = {
    "observe",
    "measure",
    "report",
    "find",
    "obtain",
    "record",
    "simulate",
    "evaluate",
    "test",
} 

hedging_verbs_score = -0.5
HEDGING_VERBS = {
    "suggest",
    "indicate",
    "appear",
    "seem",
    "tend",
    "may",
    "might",
    "could",
} 

verb_rubric = {
    'boilerplate':{'score': boilerplate_verbs_score, 'verbs': BOILERPLATE_VERBS},
    'structural':{'score': structural_verbs_score, 'verbs': STRUCTURAL_VERBS},
    'content':{'score': content_verbs_score, 'verbs': CONTENT_VERBS},
    'relation':{'score': relation_verbs_score, 'verbs': RELATION_VERBS},
    'reporting':{'score': reporting_verbs_score, 'verbs': REPORTING_VERBS},
    'hedging':{'score': hedging_verbs_score, 'verbs': HEDGING_VERBS},
}

def text_scoring(text):
    """Score text based on presence of certain verbs."""
    doc = nlp(text)
    score = 0.0
    for category, rubric in verb_rubric.items():
        verb_score = rubric['score']
        verb_list = rubric['verbs']
        verb_list = [token.lemma_ for token in doc if str(token).lower() in verb_list]
        score += verb_score * len(verb_list)
    return score

def split_text(text,delimiters):
    """Split text by multiple delimiters."""
    delim = delimiters.pop(0)
    for delimeter in delimiters:
        text = text.replace(delimeter,delim)
    return text.split(delim)

def score_segments(text):
    """Split text into segments and score each segment."""
    delimiters = ['.',';','\n']
    segments = split_text(text,delimiters)
    scored_segments = [(segment, text_scoring(segment)) for segment in segments]
    return scored_segments

def get_top_scoring_segments(text, top_k=0):
    """Get top k scoring segments from text."""
    scored_segments = score_segments(text)
    scored_segments.sort(key=lambda x: x[1], reverse=True)
    if top_k <= 0:
        return scored_segments
    return scored_segments[:top_k]

def get_top_scoring_segments_as_string(text, top_k=None,char_limit=1000):
    """Get top k scoring segments from text as a single string."""
    if top_k is None:
        top_k = 0
    top_segments = get_top_scoring_segments(text, top_k)
    result = ''
    while len(result) < char_limit and top_segments:
        segment, score = top_segments.pop(0)
        if len(result) + len(segment) + 1 <= char_limit:
            result += segment + '|'
        else:
            break
    return result.strip()