import streamlit as st
from openai import OpenAI
import os
from datetime import datetime
from fuzzywuzzy import fuzz  # 📦 Ensure installed via: pip install fuzzywuzzy

# ✅ Load the API key securely
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = "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"

# ✅ Initialize OpenAI client
client = OpenAI(api_key=api_key)

# 🎯 Streamlit App UI
st.set_page_config(page_title="Hallucination Correction Assistant", page_icon="🧠")
st.image("edfeadb0-5460-4e79-8bde-8070a0424f9c.png", width=100)
st.title("🧠 Hallucination Correction Assistant")

st.markdown("""
Welcome to the **Hallucination Correction Assistant**!  
This app uses a fine-tuned GPT-3.5 Turbo model to detect and correct AI-generated hallucinations.  
Now featuring **confidence scoring** and **cross-model comparison**.
""")

# 📝 Input box
user_input = st.text_area("✏️ Enter a potentially hallucinated response:", height=150)
comparison_model = st.selectbox("🔁 Compare with another model:", ["gpt-3.5-turbo", "gpt-4"])

# 🕒 Timestamp
st.caption(f"🕒 Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("🔍 Check & Compare") and user_input.strip():
    try:
        with st.spinner("🔍 Analyzing with fine-tuned model..."):
            response_ft = client.chat.completions.create(
                model="ft:gpt-3.5-turbo-0125:dares-apis:hallucination-logical-v1:BvwmRqwc",
                messages=[
                    {"role": "system", "content": "You are a hallucination correction assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            corrected = response_ft.choices[0].message.content

        with st.spinner(f"🤖 Getting response from {comparison_model}..."):
            response_cmp = client.chat.completions.create(
                model=comparison_model,
                messages=[
                    {"role": "system", "content": "You are a hallucination correction assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            comparison_output = response_cmp.choices[0].message.content

        # ✅ Show fine-tuned model's output
        st.subheader("✅ Corrected Output (Fine-Tuned Model)")
        st.success(corrected)
        st.code(corrected, language='markdown')

        # 🔁 Show comparison output
        st.subheader(f"🧠 Output from {comparison_model}")
        st.info(comparison_output)

        # 📊 Similarity confidence (cross-model)
        similarity = fuzz.token_set_ratio(corrected, comparison_output)
        st.metric(label=f"🔁 Similarity to {comparison_model}", value=f"{similarity}%")

        # 📈 Estimated confidence of fine-tuned output
        confidence_estimate = fuzz.token_set_ratio(user_input, corrected)
        st.metric(label="🔐 Model Confidence Estimate", value=f"{confidence_estimate}%", delta_color="normal")

        # 🧠 Explanation of confidence metrics
        with st.expander("ℹ️ What do the confidence scores mean?"):
            st.markdown("""
            - **Similarity to Model:** How similar the fine-tuned model's correction is to another model's response.
            - **Model Confidence Estimate:** How consistent the corrected output is compared to your original input.  
              Higher % = more overlap in structure and vocabulary.
            """)

    except Exception as e:
        st.error(f"🚨 An error occurred: {e}")

# 🙏 Footer
st.markdown("""
---
✅ Made with [Streamlit](https://streamlit.io/) and [OpenAI GPT-3.5 Turbo](https://platform.openai.com/docs/models/gpt-3-5).  
For educational and demonstration use only.
""")
