import streamlit as st
import generation

chain  = generation.rag_chain


def generate_response(text):
    response = chain.invoke(text)
    return response

st.title("Dinesh")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hi I am Dinesh, your ai fitness trainer How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Hi I am your fitness trainer, how can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})