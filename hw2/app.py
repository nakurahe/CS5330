"""
🤖 vs 👤 AI Face Detection Game
A fun interactive game where humans compete against AI to detect AI-generated faces!
"""

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random

# ============================================================
# Configuration and Setup
# ============================================================

# Load the trained model
MODEL_PATH = "best_mobilenet_finetuned.keras"
if os.path.exists(MODEL_PATH):
    # Load model without recompiling to avoid shape mismatch issues
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}. Please ensure the model file exists.")
    model = None

# Load test images
TEST_IMAGES_DIR = "test_images"
image_files = []
image_labels = {}

if os.path.exists(TEST_IMAGES_DIR):
    # Load real images
    real_dir = os.path.join(TEST_IMAGES_DIR, "real")
    if os.path.exists(real_dir):
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.extend(real_images)
        for img in real_images:
            image_labels[img] = 1  # 1 = Real
    
    # Load AI-generated images
    ai_dir = os.path.join(TEST_IMAGES_DIR, "ai")
    if os.path.exists(ai_dir):
        ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.extend(ai_images)
        for img in ai_images:
            image_labels[img] = 0  # 0 = AI
    
    random.shuffle(image_files)
    print(f"✅ Loaded {len(image_files)} test images")
else:
    print(f"⚠️ Test images directory not found at {TEST_IMAGES_DIR}")

# Game state
class GameState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.current_index = 0
        self.human_correct = 0
        self.human_total = 0
        self.ai_correct = 0
        self.ai_total = 0
        self.current_image = None
        self.ai_prediction = -1
        self.human_prediction = -1
        self.true_label = -1
        self.game_started = False
        self.max_rounds = 10  # Default number of rounds
        self.current_round = 0
        self.ai_results = []  # Track individual round results for AI
        self.human_results = []  # Track individual round results for human

game_state = GameState()

# ============================================================
# Helper Functions
# ============================================================

def preprocess_image(img_path, target_size=(128, 128)):
    """Preprocess image for model prediction"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Check if model expects [0, 255] (MobileNet) or [0, 1] (CNN)
    # For MobileNet models, scale back to [0, 255]
    if model and 'mobilenet' in MODEL_PATH.lower():
        img_array = img_array * 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_ai_prediction(img_path):
    """Get AI model's prediction"""
    if model is None:
        # Fallback: random prediction if model not loaded
        return random.random()
    
    try:
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"Error predicting: {e}")
        return 0.5

def create_horizontal_progress_html(player, results, max_rounds, winner_emoji=""):
    """Create HTML for horizontal progress bar with colored blocks"""
    player_icon = "🤖" if player == "AI" else "👤"
    
    # Create blocks for each round
    blocks_html = ""
    for i in range(max_rounds):
        if i < len(results):
            # Color based on result: green for correct, red for wrong
            color = "#4caf50" if results[i] else "#f44336"
            blocks_html += f'<div class="progress-block" style="background-color: {color};"></div>'
        else:
            # Gray for unplayed rounds
            blocks_html += f'<div class="progress-block" style="background-color: #e0e0e0;"></div>'
    
    # Calculate stats
    correct = sum(results)
    total = len(results)
    accuracy_text = f"{correct}/{total} ({(correct/total*100):.1f}%)" if total > 0 else "0/0 (0%)"
    
    return f"""
        <div class="horizontal-progress-container">
            <div class="progress-label">{player_icon} {player} {winner_emoji}</div>
            <div class="horizontal-progress-bar">
                {blocks_html}
            </div>
            <div class="progress-stats">{accuracy_text}</div>
        </div>
    """

# ============================================================
# Game Logic Functions
# ============================================================

def start_game(rounds):
    """Initialize a new game with specified number of rounds"""
    game_state.reset()
    
    # Validate rounds input
    if rounds is None or rounds < 1:
        rounds = 10  # Default to 10 rounds
    elif rounds > 100:
        rounds = 100  # Cap at 100 rounds
    
    game_state.max_rounds = int(rounds)
    
    if not image_files:
        return (
            gr.update(visible=False),  # rounds_row
            gr.update(visible=False),  # game_row
            gr.update(visible=False),  # buttons_row
            gr.update(visible=False),  # next_btn
            "⚠️ No test images found. Please add images to the test_images directory.",  # chat_feedback_display
            None,  # image_display
            "",  # ai_progress
            ""   # human_progress
        )
    
    game_state.game_started = True
    game_state.current_round = 1
    game_state.current_image = image_files[game_state.current_index]
    game_state.true_label = image_labels.get(game_state.current_image, -1)
    
    ai_progress_html = create_horizontal_progress_html("AI", [], game_state.max_rounds)
    human_progress_html = create_horizontal_progress_html("Human", [], game_state.max_rounds)
    
    return (
        gr.update(visible=False),  # rounds_row
        gr.update(visible=True),   # game_row
        gr.update(visible=True),   # buttons_row
        gr.update(visible=False),  # next_btn
        "**Round 1 of " + str(game_state.max_rounds) + "**\n\n🎯 **Choose:** Is this face Real or AI-Generated?",  # chat_feedback_display
        game_state.current_image,  # image_display
        ai_progress_html,          # ai_progress
        human_progress_html        # human_progress
    )

def make_prediction(choice):
    """Handle user's prediction and reveal results"""
    if not game_state.game_started or game_state.current_image is None:
        return (
            gr.update(visible=False),  # rounds_row
            gr.update(visible=False),  # game_row
            gr.update(visible=False),  # buttons_row
            gr.update(visible=False),  # next_btn
            "⚠️ Please start the game first!",  # chat_feedback_display
            None,  # image_display
            "",  # ai_progress
            ""   # human_progress
        )
    
    # Get user's choice
    game_state.human_prediction = 1 if choice == "Real" else 0
    game_state.human_total += 1
    
    # Get AI's prediction
    ai_prob = get_ai_prediction(game_state.current_image)
    game_state.ai_prediction = 1 if ai_prob > 0.5 else 0
    game_state.ai_total += 1
    
    # Check correctness
    human_correct = (game_state.human_prediction == game_state.true_label)
    ai_correct = (game_state.ai_prediction == game_state.true_label)
    
    # Track individual results
    game_state.human_results.append(human_correct)
    game_state.ai_results.append(ai_correct)
    
    if human_correct:
        game_state.human_correct += 1
    if ai_correct:
        game_state.ai_correct += 1
    
    # Calculate accuracies
    human_acc = (game_state.human_correct / game_state.human_total * 100) if game_state.human_total > 0 else 100
    ai_acc = (game_state.ai_correct / game_state.ai_total * 100) if game_state.ai_total > 0 else 100
    
    # Create result message
    true_label_str = "**Real**" if game_state.true_label == 1 else "**AI-Generated**"
    human_result = "✅ Correct!" if human_correct else "❌ Wrong!"
    ai_result = "✅ Correct!" if ai_correct else "❌ Wrong!"
    
    chat_history = f"""**Round {game_state.current_round} of {game_state.max_rounds}**

🤖 **AI**: _{("Real" if game_state.ai_prediction == 1 else "AI-Generated")}_ (confidence: {ai_prob:.1%}) {ai_result}

👤 **You**: _{choice}_ {human_result}

---
**Truth**: This face is {true_label_str}
---"""
    
    # Update progress bars
    ai_progress_html = create_horizontal_progress_html("AI", game_state.ai_results, game_state.max_rounds)
    human_progress_html = create_horizontal_progress_html("Human", game_state.human_results, game_state.max_rounds)
    
    # Feedback message
    if game_state.human_total >= 3:
        if human_acc > ai_acc:
            feedback = f"🎉 **Amazing!** You're beating AI! ({human_acc:.1f}% vs {ai_acc:.1f}%)"
        elif human_acc < ai_acc:
            feedback = f"🤖 **AI is winning!** Keep trying! ({ai_acc:.1f}% vs {human_acc:.1f}%)"
        else:
            feedback = f"🤝 **It's a tie!** You're matching AI! ({human_acc:.1f}%)"
    else:
        feedback = ""
    
    # Combine chat history and feedback
    combined_content = chat_history
    if feedback:
        combined_content += f"\n\n{feedback}"
    
    # Check if this is the last round
    is_last_round = (game_state.current_round >= game_state.max_rounds or 
                     game_state.current_index >= len(image_files) - 1)
    
    if is_last_round:
        # This is the last round, show final results directly
        return next_round()
    
    return (
        gr.update(visible=False),  # rounds_row
        gr.update(visible=True),   # game_row
        gr.update(visible=False),  # buttons_row
        gr.update(visible=True),   # next_btn
        combined_content,          # chat_feedback_display
        game_state.current_image,  # image_display
        ai_progress_html,          # ai_progress
        human_progress_html        # human_progress
    )

def next_round():
    """Load next image"""
    game_state.current_index += 1
    game_state.current_round += 1
    
    if game_state.current_round > game_state.max_rounds or game_state.current_index >= len(image_files):
        # End of game
        human_acc = (game_state.human_correct / game_state.human_total * 100) if game_state.human_total > 0 else 0
        ai_acc = (game_state.ai_correct / game_state.ai_total * 100) if game_state.ai_total > 0 else 0
        
        # Determine winner emojis
        if human_acc > ai_acc:
            final_msg = f"🎊 **CONGRATULATIONS!** 🎊\n\nYou beat AI!\n\n👤 You: {human_acc:.1f}%\n🤖 AI: {ai_acc:.1f}%"
            human_emoji = "👑"
            ai_emoji = "😔"
        elif human_acc < ai_acc:
            final_msg = f"🤖 **AI WINS!** 🤖\n\nAI performed better this time!\n\n🤖 AI: {ai_acc:.1f}%\n👤 You: {human_acc:.1f}%"
            ai_emoji = "👑"
            human_emoji = "😔"
        else:
            final_msg = f"🤝 **IT'S A TIE!** 🤝\n\nYou matched AI perfectly!\n\n👤 You: {human_acc:.1f}%\n🤖 AI: {ai_acc:.1f}%"
            ai_emoji = "🤝"
            human_emoji = "🤝"
        
        final_msg += f"\n\n_Completed {game_state.max_rounds} rounds! Click 'Start New Game' to play again!_"
        
        # Update progress bars with final results and winner emojis
        ai_progress_html = create_horizontal_progress_html("AI", game_state.ai_results, game_state.max_rounds, ai_emoji)
        human_progress_html = create_horizontal_progress_html("Human", game_state.human_results, game_state.max_rounds, human_emoji)
        
        return (
            gr.update(visible=True),   # rounds_row (show again for new game)
            gr.update(visible=True),   # game_row
            gr.update(visible=False),  # buttons_row
            gr.update(visible=False),  # next_btn
            final_msg,                 # chat_feedback_display
            game_state.current_image,  # image_display
            ai_progress_html,          # ai_progress
            human_progress_html        # human_progress
        )
    
    # Load next image
    game_state.current_image = image_files[game_state.current_index]
    game_state.true_label = image_labels.get(game_state.current_image, -1)
    game_state.human_prediction = -1
    game_state.ai_prediction = -1

    # Update progress bars
    ai_progress_html = create_horizontal_progress_html("AI", game_state.ai_results, game_state.max_rounds)
    human_progress_html = create_horizontal_progress_html("Human", game_state.human_results, game_state.max_rounds)

    return (
        gr.update(visible=False),  # rounds_row
        gr.update(visible=True),   # game_row
        gr.update(visible=True),   # buttons_row
        gr.update(visible=False),  # next_btn
        f"🎯 **Choose:** Is this face Real or AI-Generated?\n\n*Round {game_state.current_round} of {game_state.max_rounds}*",  # chat_feedback_display
        game_state.current_image,  # image_display
        ai_progress_html,          # ai_progress
        human_progress_html        # human_progress
    )

# ============================================================
# Gradio Interface
# ============================================================

# Custom CSS for better styling
custom_css = """
.score-box {
    font-size: 1.2em;
    font-weight: bold;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    background: linear-gradient(135deg, #81c784 0%, #66bb6a 100%);
    color: white;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
}
.game-title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.chat-box {
    min-height: 200px;
    padding: 15px;
    border-radius: 10px;
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
    font-size: 1.1em;
    border: 2px solid #a5d6a7;
}
.feedback-box {
    font-size: 1.3em;
    font-weight: bold;
    padding: 10px;
    text-align: center;
    border-radius: 8px;
    background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
    color: #2e7d32;
    min-height: 60px;
    margin-bottom: 10px;
}
.horizontal-progress-container {
    margin: 10px 0;
    padding: 10px;
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e8 100%);
    border-radius: 8px;
    border: 2px solid #a5d6a7;
}
.horizontal-progress-bar {
    display: flex;
    gap: 2px;
    margin: 8px 0;
    justify-content: center;
}
.progress-block {
    width: 20px;
    height: 20px;
    border-radius: 3px;
    border: 1px solid #ccc;
    flex-shrink: 0;
}
.progress-label {
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    color: #2e7d32;
    margin: 5px 0;
}
.progress-stats {
    font-size: 14px;
    text-align: center;
    color: #2e7d32;
    margin: 5px 0;
}
/* Additional green styling for buttons and components */
.gradio-button {
    background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%) !important;
    border-color: #4caf50 !important;
}
.gradio-button:hover {
    background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%) !important;
}
/* Specific styling for primary buttons (Start Game, Next Round) */
button[variant="primary"] {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%) !important;
    border-color: #2e7d32 !important;
    color: white !important;
    font-weight: bold !important;
}
button[variant="primary"]:hover {
    background: linear-gradient(135deg, #388e3c 0%, #1b5e20 100%) !important;
    border-color: #1b5e20 !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(r"""
    <pre style="font-family: monospace; font-size: 14px; line-height: 1.2;">
      🤖 AI vs HUMAN 👤
    ╭─────────────────────────╮   ___________________________________
    │    ┌─┐     ┌───┐        │  /                                   \
    │    │●│ VS  │ ◉ │        │ |  Can you outsmart the AI at        |
    │    └─┘     └───┘        │ |  detecting AI-generated faces?     |
    │   ROBOT    HUMAN        │ |  Let's find out!                   |
    │                         │  \___________________________________/
    ╰─────────────────────────╯
    </pre>
    """)
    
    # First row: Number of rounds and start button
    with gr.Row() as rounds_row:
        with gr.Column(scale=2):
            with gr.Row():
                gr.Markdown("**🎯Choose Rounds:**", elem_classes="progress-label")
                rounds_input = gr.Number(
                    value=10,
                    minimum=1,
                    maximum=100,
                    step=1,
                    show_label=False,
                    container=False,
                    scale=0
                )
                gr.Markdown("*(1-100 rounds)*", elem_classes="progress-stats")
        with gr.Column(scale=1):
            start_btn = gr.Button("Start New Game", variant="primary", size="lg")
    
    # Second row: Image display on left, feedback and progress bars on right
    with gr.Row(visible=False) as game_row:
        with gr.Column(scale=2):
            # Image display
            image_display = gr.Image(
                label="Can you tell if this face is real or AI-generated?",
                type="filepath",
                height=600
            )
        
        with gr.Column(scale=1):
            # # Combined chat and feedback area
            chat_feedback_display = gr.Markdown(
                value="",
                elem_classes="chat-box"
            )
            
            # AI Progress Bar
            ai_progress = gr.HTML("""
                <div class="horizontal-progress-container">
                    <div class="progress-label">🤖 AI</div>
                    <div class="horizontal-progress-bar"></div>
                    <div class="progress-stats">0/0 (0%)</div>
                </div>
            """)
            
            # Human Progress Bar
            human_progress = gr.HTML("""
                <div class="horizontal-progress-container">
                    <div class="progress-label">👤 Human</div>
                    <div class="horizontal-progress-bar"></div>
                    <div class="progress-stats">0/0 (0%)</div>
                </div>
            """)
    
    # Third row: Centered AI and Real buttons
    with gr.Row(visible=False) as buttons_row:
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1):
            with gr.Row():
                ai_btn = gr.Button("AI-Generated", variant="secondary", size="lg")
                real_btn = gr.Button("Real", variant="secondary", size="lg")
        with gr.Column(scale=1):
            pass
    
    # Next button (appears when needed)
    next_btn = gr.Button("Next Round", variant="primary", size="lg", visible=False)
    
    gr.Markdown("""
    ---
    ### 📊 About This Game:
    This interactive game uses a deep learning model trained to distinguish between real human faces and AI-generated faces.
    The model has been trained on 2000 images and has an accuracy of over 70%.
    
    """)
    
    # Event handlers
    start_btn.click(
        fn=start_game,
        inputs=[rounds_input],
        outputs=[rounds_row, game_row, buttons_row, next_btn, chat_feedback_display, image_display, ai_progress, human_progress]
    )
    
    ai_btn.click(
        fn=lambda: make_prediction("AI-Generated"),
        inputs=[],
        outputs=[rounds_row, game_row, buttons_row, next_btn, chat_feedback_display, image_display, ai_progress, human_progress]
    )
    
    real_btn.click(
        fn=lambda: make_prediction("Real"),
        inputs=[],
        outputs=[rounds_row, game_row, buttons_row, next_btn, chat_feedback_display, image_display, ai_progress, human_progress]
    )
    
    next_btn.click(
        fn=next_round,
        inputs=[],
        outputs=[rounds_row, game_row, buttons_row, next_btn, chat_feedback_display, image_display, ai_progress, human_progress]
    )

# ============================================================
# Launch the app
# ============================================================

if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allows external connections
        server_port=15  # Default Gradio port
    )
