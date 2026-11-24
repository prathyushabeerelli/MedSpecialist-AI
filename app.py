import streamlit as st
import torch
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="MedSpecialist AI", page_icon="🩺", layout="centered")

# Replace this with your public Hugging Face model repository ID.
MODEL_REPO = "YOUR-HUGGINGFACE-USERNAME/medspecialist-model"

@st.cache_resource(show_spinner="Loading MedSpecialist AI model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
    labels_path = hf_hub_download(repo_id=MODEL_REPO, filename="label_classes.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    model.eval()
    return tokenizer, model, class_names

st.title("MedSpecialist AI")
st.caption("Medical transcription specialty classification using Bio_ClinicalBERT")
st.info("Academic demonstration only. This tool is not a medical diagnosis system and must not be used for clinical decisions.")

with st.form("prediction_form"):
    sample_name = st.text_input("Sample name / context (optional)")
    description = st.text_area("Summary / description (optional)", height=90)
    keywords = st.text_input("Keywords (optional)")
    transcription = st.text_area("Medical transcription", height=240, placeholder="Paste the medical transcription here...")
    submitted = st.form_submit_button("Predict medical specialty", type="primary")

if submitted:
    if not transcription.strip():
        st.warning("Please enter a medical transcription.")
    elif MODEL_REPO.startswith("YOUR-"):
        st.error("Set MODEL_REPO in app.py to your Hugging Face model repository ID.")
    else:
        try:
            tokenizer, model, class_names = load_model()
            combined_text = (
                "Specialty Context: " + sample_name.strip() +
                " | Summary: " + description.strip() +
                " | Keywords: " + keywords.strip() +
                " | Note: " + transcription.strip()
            )
            # Match the notebook's Head+Tail token handling (380 first + 130 last).
            tokens = tokenizer.encode(combined_text, add_special_tokens=False)
            if len(tokens) > 510:
                tokens = tokens[:380] + tokens[-130:]
            input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            pad_len = 512 - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
            with torch.inference_mode():
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)[0]
            confidence, predicted_id = torch.max(probabilities, dim=0)
            predicted_label = class_names[int(predicted_id)]
            st.success(f"Predicted specialty: {predicted_label}")
            st.metric("Model confidence", f"{confidence.item() * 100:.2f}%")
            st.subheader("Top 3 predictions")
            top_probs, top_ids = torch.topk(probabilities, k=min(3, len(probabilities)))
            for prob, idx in zip(top_probs, top_ids):
                label = class_names[int(idx)]
                st.write(f"**{label}** — {prob.item() * 100:.2f}%")
                st.progress(float(prob.item()))
        except Exception as e:
            st.error("The model could not be loaded or prediction failed. Check the model repository ID and uploaded model files.")
            st.exception(e)
