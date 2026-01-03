import gradio as gr
import uuid
from core.src.rag_workflow import RagChatWorkflow

rag_chat = RagChatWorkflow()

async def chat_handler(message, history, user_name, user_id):
    active_id = user_id if user_id.strip() else "guest_user"
    
    rag_response = await rag_chat.run(
        user_query=message,
        user_name=user_name, 
        user_id=active_id
    )
    return str(rag_response)

# 1. Removed theme from Blocks constructor
with gr.Blocks() as demo:
    gr.Markdown("# My RAG Assistant")
    
    with gr.Row():
        name_input = gr.Textbox(label="User Name", value="Explorer")
        id_input = gr.Textbox(label="Session/User ID", placeholder="Enter ID...")

    # 2. Removed type="messages" (it's now the default)
    chat_ui = gr.ChatInterface(
        fn=chat_handler,
        additional_inputs=[name_input, id_input]
    )

if __name__ == "__main__":
    # 3. Moved theme to launch() as suggested by the warning
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft()
    )