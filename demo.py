from nubia import Nubia
import gradio

nubia = Nubia()


def predict(inp_1, inp_2):
    features = nubia.score(inp_1, inp_2, get_features=True)
    labels = {k: v for k, v in features["features"].items()}
    labels["Likelihood Contradiction"] = labels["contradiction"]
    del labels["contradiction"]
    labels["Likelihood logical agreeement"] = labels["logical_agreement"]
    del labels["logical_agreement"]
    labels["Likelihood of irrelevancy (or new information)"] = labels["irrelevancy"]
    del labels["irrelevancy"]
    labels["Semantic Similarity (out of 5.0)"] = labels["semantic_relation"]
    del labels["semantic_relation"]
    del labels["grammar_hyp"]
    del labels["grammar_ref"]
    return {"nubia_score": features["nubia_score"]}, labels


title = "NUBIA"
description = "NUBIA stands for 'NeUral Based Interchangeability Assessor' \n NUBIA gives a score on a scale of 0 to 1 reflecting how much it thinks the candidate text is interchangeable with the reference text. It also shows its rationale for the score."
inputs = [gradio.inputs.Textbox(lines=5, label="Reference Text"), gradio.inputs.Textbox(lines=5, label="Candidate Text")]
outputs = [gradio.outputs.Label(label="Interchangeability Score"), gradio.outputs.JSON(label="Neural Features (Explanation)")]
examples = [
    ["This car is expensive! I can't buy it.", "That automobile costs a fortune! Purchasing it? Impossible!"],
    ["This car is expensive! I can't buy it.", "That automobile costs a good amount. Purchasing it? Totally feasible!"],
    ["The dinner was delicious.", "The dinner did not taste good."]
]
iface = gradio.Interface(fn=predict, inputs=inputs, outputs=outputs, capture_session=True, examples=examples,
                         title=title, description=description, allow_flagging=False)

iface.launch()
