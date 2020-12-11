from nubia import Nubia
import gradio

nubia = Nubia()


def predict(inp_1, inp_2):
    features = nubia.score(inp_1, inp_2, get_features=True)
    labels = {k: v for k, v in features["features"].items()}
    friendly_keys = [
    "Percentage Contradiction",
    "Perplexity Hypothesis Sentence", 
    "Perplexity Reference Sentence",
    "Percentage irrelevancy or new information",
    "Percentage logical agreement",
    "Semantic Relationship (out 5.0)"
    ]
    labels = dict(zip(friendly_keys, list(labels.values()))) 
    return {"nubia_score": features["nubia_score"]}, labels


title = "NUBIA"
description = "NeUral Based Interchangeability Assessor. NUBIA gives a score on a scale of 0 to 1 reflecting how much it thinks the two sentences are interchangeable"
inputs = [gradio.inputs.Textbox(lines=5, label="Reference Text"), gradio.inputs.Textbox(lines=5, label="Candidate Text")]
outputs = [gradio.outputs.Label(label="Interchangeability Score"), gradio.outputs.JSON(label="All Features")]
examples = [
    ["This car is expensive! I can't buy it.", "That automobile costs a fortune! Purchasing it? Impossible!"],
    ["This car is expensive! I can't buy it.", "That automobile costs a good amount. Purchasing it? Totally feasible!"],
    ["The dinner was delicious.", "The dinner did not taste good."]
]
iface = gradio.Interface(fn=predict, inputs=inputs, outputs=outputs, capture_session=True, examples=examples,
                         title=title, description=description, allow_flagging=False)

iface.launch()
