import streamlit as st
import tensorflow as tf
import numpy as np
import trax

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

# Streamlit UI
st.title("🌍 ML Project: English-to-German Translation with Seq2Seq Transformer")
st.markdown("Translate English sentences to German using a Transformer with an Attention Mechanism.")

st.sidebar.header("Translation Settings")
st.sidebar.write("Configure translation options")

# Input Section
st.subheader("1. Input your English sentence")
input_text = st.text_area("Enter the sentence:", "It is nice to learn new things today!", height=70)

# Translation and Display Section
if st.button("Translate to German"):
    if input_text:
        st.write("🔄 **Translating...**")
        
        # Initialize and run the model for translation
        model = trax.models.Transformer(
            input_vocab_size=33300, d_model=512, d_ff=2048,
            n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
            max_len=2048, mode='predict')
        
        model.init_from_file('./outputs/ende_wmt32k.pkl.gz', weights_only=True)
        
        # Preprocess input and predict translation
        tokenized = preprocess_text(input_text)
        tokenized_translation = trax.supervised.decoding.autoregressive_sample(model, tokenized, temperature=0.0)
        
        # Postprocess and display translation
        translation = postprocess_output(tokenized_translation)
        
        # Display Results with Markdown styling
        st.subheader("2. Translation Output")
        st.markdown("### Translated Sentence:")
        st.success(translation)
    else:
        st.warning("⚠️ Please enter a sentence to translate.")
        
# Optional Information for Users
st.sidebar.subheader("Project Details")
st.sidebar.markdown("""
- **Model**: Transformer (6 layers)
- **Vocabulary Size**: 33,300
- **Architecture**: Seq2Seq with Attention
- **Dataset**: WMT-32K English-German Translation Dataset
""")
st.sidebar.write("🔍 **Note:** Higher temperature values yield more diverse translations.")
