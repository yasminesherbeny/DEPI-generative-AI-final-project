"""
streamlit_app.py
=================
User interface for the DEPI Generative AI Final Project — Automated Product
Description Generator.

A thin client: all the work (RAG retrieval + LoRA-fine-tuned GPT-2 generation)
happens behind the Flask API in app.py. This app just collects product info,
calls POST /generate, and renders the result.

Run the API first (from the machine/service hosting it):
    python app.py

Then run this app:
    streamlit run streamlit_app.py
"""
import os
from datetime import datetime

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("PRODUCT_API_URL", "http://localhost:7860")

st.set_page_config(
    page_title="Product Description Generator",
    page_icon="🛍️",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []


def call_api(api_url: str, name: str, color: str, category: str, top_k: int) -> dict:
    response = requests.post(
        f"{api_url.rstrip('/')}/generate",
        json={"name": name, "color": color, "category": category, "top_k": top_k},
        timeout=60,
    )
    payload = response.json()
    if response.status_code != 200:
        raise RuntimeError(payload.get("error", f"API returned status {response.status_code}"))
    return payload


def check_api(api_url: str) -> bool:
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------- Sidebar: API connection ----------

with st.sidebar:
    st.header("API Connection")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)

    if check_api(api_url):
        st.success("API is reachable")
    else:
        st.error("API is not reachable")
        st.caption("Start it with `python app.py` on the machine hosting the model, then refresh.")

    st.divider()

# ---------- Main ----------

st.title("🛍️ Automated Product Description Generator")
st.caption("DEPI Generative AI Final Project — GPT-2 fine-tuned with LoRA, enhanced with RAG retrieval over similar products.")

with st.form("product_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Product name *", placeholder="e.g. Running Shoes")
    with col2:
        color = st.text_input("Color", placeholder="e.g. Black")
    with col3:
        category = st.text_input("Category", placeholder="e.g. Footwear")

    top_k = st.slider("Similar products to retrieve", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Generate Description", type="primary")

if submitted:
    if not name.strip():
        st.warning("Product name is required.")
    else:
        with st.spinner("Retrieving similar products and generating description..."):
            try:
                result = call_api(api_url, name.strip(), color.strip(), category.strip(), top_k)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Couldn't connect to the API. Make sure it's running "
                    f"(`python app.py`) and reachable at {api_url}."
                )
            except RuntimeError as e:
                st.error(f"Generation failed: {e}")
            else:
                st.session_state.history.append({
                    "name": result["name"],
                    "description": result["description"],
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                })

                st.subheader("Generated Description")
                st.markdown(f"> {result['description']}")

                similar = result.get("similar_products", [])
                st.subheader(f"Similar Products ({len(similar)})")
                if similar:
                    st.dataframe(
                        [
                            {
                                "Name": p["name"],
                                "Color": p["color"],
                                "Similarity Score": p["score"],
                            }
                            for p in similar
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No similar products found for these filters.")

# ---------- Sidebar: history (rendered after the form logic above so a
# generation from this run shows up immediately, not on the next rerun) ----------

with st.sidebar:
    st.header("History")
    if not st.session_state.history:
        st.caption("Generated descriptions will appear here.")
    else:
        for item in reversed(st.session_state.history[-10:]):
            with st.expander(f"{item['name']} · {item['timestamp']}"):
                st.write(item["description"])
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
