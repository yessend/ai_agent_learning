import gradio as gr
import uuid
from core.src.rag.rag_workflow import RagChatWorkflow

rag_chat = RagChatWorkflow()

USERS = {"sanzhar": "admin123", "sister": "math123"}

# --- RAG Logic ---
async def chat_handler(message, history, user_name):
    """
    message: the current user prompt
    history: the conversation history (provided by ChatInterface)
    user_name: the name stored in gr.State after login
    """
    # If for some reason state is empty, fallback to guest
    active_id = user_name if user_name else "guest_user"
    
    # This calls your actual RAG backend
    # Note: We await it because your handler is async
    rag_response = await rag_chat.run(
        user_query=message,
        user_name=user_name, 
        user_id=active_id
    )
    return str(rag_response)

# --- Auth Actions ---
def register_action(new_user, new_pass):
    if not new_user or not new_pass:
        return "⚠️ Please fill in both fields."
    if new_user in USERS:
        return "❌ Username already exists."
    USERS[new_user] = new_pass
    return f"✅ Account created for {new_user}! You can now login."

def login_action(username, password):
    if USERS.get(username) == password:
        # 1. Hide Auth Page, 2. Show App Page, 3. Update Status, 4. Update gr.State
        return (
            gr.update(visible=False), 
            gr.update(visible=True), 
            f"Logged in as: **{username}**",
            username # This sets the current_user State
        )
    return gr.update(visible=True), gr.update(visible=False), "❌ Invalid login.", ""

# --- UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # This keeps track of the user across the session
    current_user = gr.State("")

    # PAGE 1: AUTH (Login & Sign Up)
    with gr.Column(visible=True) as auth_page:
        gr.Markdown("# 🔐 Access Portal")
        with gr.Tabs():
            with gr.TabItem("Login"):
                u_login = gr.Textbox(label="Username")
                p_login = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Login", variant="primary")
                login_msg = gr.Markdown()

            with gr.TabItem("Sign Up"):
                u_reg = gr.Textbox(label="New Username")
                p_reg = gr.Textbox(label="New Password", type="password")
                reg_btn = gr.Button("Create Account")
                reg_msg = gr.Markdown()

    # PAGE 2: THE APP (RAG UI)
    with gr.Column(visible=False) as app_page:
        status_label = gr.Markdown()
        
        # We pass current_user as an additional input so chat_handler gets it
        gr.ChatInterface(
            fn=chat_handler,
            additional_inputs=[current_user]
        )

    # --- Button Logic ---
    reg_btn.click(register_action, [u_reg, p_reg], reg_msg)
    
    login_btn.click(
        login_action, 
        [u_login, p_login], 
        [auth_page, app_page, status_label, current_user]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)