from nubia import Nubia
import gradio

nubia = Nubia()


def predict(inp_1, inp_2):
    features = nubia.score(inp_1, inp_2, get_features=True)
    labels = {k: v for k, v in features["features"].items()}
    feature_dict = {}
    feature_dict["Semantic Relationship (out 5.0)"]=labels["semantic_relation"]
    feature_dict["Percentage logical agreement"]=labels["logical_agreement"]
    feature_dict["Percentage irrelevancy or new information"]= labels["irrelevancy"]
    labels = dict(zip(friendly_keys, list(labels.values()))) 
    return {"nubia_score": features["nubia_score"]}, feature_dict


title = "NUBIA"
description = "NeUral Based Interchangeability Assessor. \n NUBIA gives a score on a scale of 0 to 1 reflecting how much it thinks the reference and candidate sentences are interchangeable"
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
