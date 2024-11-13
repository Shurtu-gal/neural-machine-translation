import streamlit as st

st.set_page_config(page_title="English-to-German Translator", page_icon="🌍")
st.title("🌍 Machine Learning Lab Project")
st.markdown("Translate English sentences to German using a Transformer model with an Attention Mechanism.")

st.sidebar.header("Translation Settings")
temperature = st.sidebar.slider("Diversity Temperature", 0.0, 1.0, 0.0, step=0.1)
st.sidebar.markdown("""
Adjusting the temperature can yield more diverse translations.
- **0.0**: Deterministic (most confident predictions)
- **1.0**: Maximum diversity
""")

import tensorflow as tf
import numpy as np
import trax
import time

# Helper functions for processing
def preprocess_text(sentence):
    return list(trax.data.tokenize(iter([sentence]),vocab_dir='./outputs/',vocab_file='ende_32k.subword'))[0][None, :]

def postprocess_output(output_ids):
    tokenized_translation = output_ids[0][:-1]  # Remove batch and EOS.
    translation = trax.data.detokenize(tokenized_translation, vocab_dir='./outputs/', vocab_file='ende_32k.subword')
    return translation

# Load the Keras model and weights
def create_keras_model(first_time=False):
    if first_time:
        return None
    model = trax.models.Transformer(input_vocab_size=33300, d_model=512, d_ff=2048, n_heads=8, 
                                    n_encoder_layers=6, n_decoder_layers=6, max_len=2048, mode='predict')
    model.init_from_file('./outputs/ende_wmt32k.pkl.gz', weights_only=True)
    keras_layer = trax.AsKeras(model, batch_size=1)
    
    # Define Keras model
    input_ids = tf.keras.Input(shape=(1024,), dtype='int32', name="input_ids")
    target_ids = tf.keras.Input(shape=(1024,), dtype='int32', name="target_ids")
    hidden = keras_layer((input_ids, target_ids))
    keras_model = tf.keras.Model(inputs=[input_ids, target_ids], outputs=hidden)
    
    return model

# Initialize model on first run
keras_model = create_keras_model(first_time=True)


model = trax.models.Transformer(
            input_vocab_size=33300, d_model=512, d_ff=2048,
            n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
            max_len=2048, mode='predict')
        
def reset_and_translate(model, sentence, temperature=0.0, progress_bar=None):
    progress_bar.progress(5)
    model.init_from_file('./outputs/ende_wmt32k.pkl.gz', weights_only=True)  # Reinitialize weights
    progress_bar.progress(30)
    tokenized = list(trax.data.tokenize(iter([sentence]), vocab_dir='./outputs/', vocab_file='ende_32k.subword'))[0]
    tokenized = tokenized[None, :]  # Add batch dimension

    progress_bar.progress(60)

    # Perform autoregressive sampling for translation
    tokenized_translation = trax.supervised.decoding.autoregressive_sample(model, tokenized, temperature=temperature)
    outputs = tokenized_translation[0][:-1]  # Remove batch and EOS

    progress_bar.progress(80)

    # Convert to text
    translated_sentence = trax.data.detokenize(outputs, vocab_dir='./outputs/', vocab_file='ende_32k.subword')

    progress_bar.progress(100)
    return translated_sentence

# Input Section
st.subheader("Enter Your Sentence in English")
input_text = st.text_area("Type the sentence you want to translate to German:", 
                          "It is nice to learn new things today!", height=70)

# Translation and Display Section
if st.button("Translate to German"):
    if input_text.strip():
        with st.spinner("Translating..."):
            progress_bar = st.progress(0)

            # Translate the sentence
            translation = reset_and_translate(model, input_text, temperature=temperature, progress_bar=progress_bar)

            progress_bar.empty()
            
            # Display Results with Markdown styling
            st.markdown("### Translated Sentence:")
            st.success(translation)

    else:
        st.warning("⚠️ Please enter a sentence to translate.")

# Sidebar: Project Details
st.sidebar.subheader("Project Details")
st.sidebar.markdown("""
- **Model**: Transformer (6 layers)
- **Vocabulary Size**: 33,300
- **Architecture**: Seq2Seq with Attention
- **Dataset**: WMT-14 English-German Translation Dataset
""")
st.sidebar.write("🔍 For more accurate translations, try lower temperatures.")

st.markdown(
    """
    <style>
    .stTextArea>div>textarea {
        font-size: 16px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
  
footer = """
    <style>
    .footer {
        position: relative;
        left: 50%;
        transform: translateX(-50%);
        bottom: 0;
        width: calc(100% - 250px);  /* Adjusts based on sidebar width */
        background-color: transparent;
        text-align: center;
        padding: 10px 0;
        padding-top: 100px;
        color: #888;
        max-width: 1000px;  /* Limits width for better alignment */
    }
    </style>
    <div class="footer">
        <hr style='border-color: #ddd;' />
        Made with ❤️ by <a href='https://github.com/shurtu-gal' target="_blank">Ashish Padhy</a>
    </div>
"""

st.markdown(footer, unsafe_allow_html=True)
