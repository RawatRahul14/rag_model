# === Modules ===
import streamlit as st
import uuid
import asyncio

from rag_model.utils.common import save_files
from rag_model.components.retriever import create_retriever
from graph import rag_graph


# === Main UI ===
def main():

    # === Session State ===
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:8]

    if "uploaded" not in st.session_state:
        st.session_state["uploaded"] = False

    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "chat_active" not in st.session_state:
        st.session_state["chat_active"] = True

    # === Heading ===
    st.header("Agentic RAG Model")

    # === Sidebar ===
    with st.sidebar:

        if not st.session_state["uploaded"]:

            with st.form("upload_form"):
                file = st.file_uploader(
                    "Upload a PDF",
                    type=["pdf"],
                    accept_multiple_files=False,
                )

                submitted = st.form_submit_button("Submit")

            if submitted:
                if file is None:
                    st.warning("Please upload a PDF first.")
                else:
                    try:
                        # === Save File ===
                        file_path = asyncio.run(
                                save_files(
                                    session_id=st.session_state["session_id"],
                                    file_bytes=file.getvalue(),
                                )
                            )

                        # === Create Retriever ===
                        retriever = create_retriever(file_path=file_path)

                        # === Store Results ===
                        st.session_state["retriever"] = retriever
                        st.session_state["uploaded"] = True

                        st.success("PDF uploaded and indexed successfully.")
                        st.rerun()

                    except Exception as exc:
                        st.error(f"Failed to process PDF: {exc}")

        else:
            st.success("Document ready for querying.")

            if st.button("End chat"):
                st.session_state["chat_active"] = False
                st.rerun()

            if st.button("Start new chat"):
                st.session_state["messages"] = []
                st.session_state["chat_active"] = True
                st.rerun()

    # === Chat Area ===
    if not st.session_state["uploaded"]:
        st.info("Upload a PDF to start chatting.")
        return

    if not st.session_state["chat_active"]:
        st.info("Chat ended. Click 'Start new chat' in the sidebar to resume.")
        return

    # === Display Chat History ===
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # === Chat Input ===
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        # === Quit Command ===
        if question.strip().lower() == "quit":
            st.session_state["chat_active"] = False
            st.rerun()

        # === Display User Message ===
        st.session_state["messages"].append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        # === Run Agentic RAG Graph ===
        try:
            result = rag_graph.invoke(
                {
                    "question": question,
                    "previous_question": (
                        st.session_state.get("previous_question", "")
                    ),
                    "rewritten_question": "",
                    "documents": [],
                    "answer": "",
                },
                config={
                    "configurable": {
                        "retriever": st.session_state["retriever"],
                    }
                },
            )

            answer = result["answer"]

            # === Store Previous Question ===
            st.session_state["previous_question"] = question

        except Exception as exc:
            answer = f"Something went wrong: {exc}"

        # === Display and Store Answer ===
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.rerun()

if __name__ == "__main__":
    main()