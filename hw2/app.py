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
MODEL_PATH = "best_mobilenet_model_for_deployment.keras"
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

def get_accuracy_color(accuracy):
    """Get color based on accuracy"""
    if accuracy >= 80:
        return "🟢"
    elif accuracy >= 60:
        return "🟡"
    else:
        return "🔴"

# ============================================================
# Game Logic Functions
# ============================================================

def start_game():
    """Initialize a new game"""
    game_state.reset()
    
    if not image_files:
        return (
            None,
            "⚠️ No test images found. Please add images to the test_images directory.",
            "🤖 **AI**: 0/0 (---%)",
            "👤 **Human**: 0/0 (---%)  ",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(visible=True)
        )
    
    game_state.game_started = True
    game_state.current_image = image_files[game_state.current_index]
    game_state.true_label = image_labels.get(game_state.current_image, -1)
    
    return (
        game_state.current_image,
        "👤 **You**: _Thinking..._\n\n🤖 **AI**: _Analyzing..._",
        "🤖 **AI**: 0/0 (100%)",
        "👤 **Human**: 0/0 (100%)",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(visible=False)
    )

def make_prediction(choice):
    """Handle user's prediction and reveal results"""
    if not game_state.game_started or game_state.current_image is None:
        return (
            "⚠️ Please start the game first!",
            "🤖 **AI**: 0/0 (100%)",
            "👤 **Human**: 0/0 (100%)",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            ""
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
    
    chat_history = f"""👤 **You**: _{choice}_ {human_result}

🤖 **AI**: _{("Real" if game_state.ai_prediction == 1 else "AI-Generated")}_ (confidence: {ai_prob:.1%}) {ai_result}

---
**Truth**: This face is {true_label_str}
---"""
    
    # Update score displays with emoji indicators
    human_emoji = get_accuracy_color(human_acc)
    ai_emoji = get_accuracy_color(ai_acc)
    
    ai_score_text = f"🤖 **AI**: {game_state.ai_correct}/{game_state.ai_total} ({ai_acc:.1f}%) {ai_emoji}"
    human_score_text = f"👤 **Human**: {game_state.human_correct}/{game_state.human_total} ({human_acc:.1f}%) {human_emoji}"
    
    # Feedback message
    if game_state.human_total >= 10:
        if human_acc > ai_acc:
            feedback = f"🎉 **Amazing!** You're beating the AI! ({human_acc:.1f}% vs {ai_acc:.1f}%)"
        elif human_acc < ai_acc:
            feedback = f"🤖 **The AI is winning!** Keep trying! ({ai_acc:.1f}% vs {human_acc:.1f}%)"
        else:
            feedback = f"🤝 **It's a tie!** You're matching the AI! ({human_acc:.1f}%)"
    else:
        feedback = ""
    
    return (
        chat_history,
        ai_score_text,
        human_score_text,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        feedback
    )

def next_round():
    """Load next image"""
    game_state.current_index += 1
    
    if game_state.current_index >= len(image_files):
        # End of game
        human_acc = (game_state.human_correct / game_state.human_total * 100) if game_state.human_total > 0 else 0
        ai_acc = (game_state.ai_correct / game_state.ai_total * 100) if game_state.ai_total > 0 else 0
        
        if human_acc > ai_acc:
            final_msg = f"🎊 **CONGRATULATIONS!** 🎊\n\nYou beat the AI!\n\n👤 You: {human_acc:.1f}%\n🤖 AI: {ai_acc:.1f}%"
        elif human_acc < ai_acc:
            final_msg = f"🤖 **AI WINS!** 🤖\n\nThe AI performed better this time!\n\n🤖 AI: {ai_acc:.1f}%\n👤 You: {human_acc:.1f}%"
        else:
            final_msg = f"🤝 **IT'S A TIE!** 🤝\n\nYou matched the AI perfectly!\n\n👤 You: {human_acc:.1f}%\n🤖 AI: {ai_acc:.1f}%"
        
        final_msg += "\n\n_Click 'Start New Game' to play again!_"
        
        return (
            game_state.current_image,
            final_msg,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            f"🎮 **Game Over!** Total rounds: {game_state.human_total}",
            gr.update(visible=True)
        )
    
    # Load next image
    game_state.current_image = image_files[game_state.current_index]
    game_state.true_label = image_labels.get(game_state.current_image, -1)
    game_state.human_prediction = -1
    game_state.ai_prediction = -1

    return (
        game_state.current_image,
        "👤 **You**: _Thinking..._\n\n🤖 **AI**: _Analyzing..._",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        gr.update(visible=False)
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
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.game-title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.chat-box {
    min-height: 200px;
    padding: 15px;
    border-radius: 10px;
    background-color: #f7f7f7;
    font-size: 1.1em;
}
.feedback-box {
    font-size: 1.3em;
    font-weight: bold;
    padding: 10px;
    text-align: center;
    border-radius: 8px;
}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    <div class="game-title">
    🤖 vs 👤 AI Face Detection Challenge
    </div>
    """)
    
    gr.Markdown("""
    ### 🎮 How to Play:
    1. Look at the face image carefully
    2. Decide if it's **Real** or **AI-Generated**
    3. Click your choice and see how you compare to the AI!
    4. Try to beat the AI's accuracy! 🏆
    """)
    
    with gr.Row():
        # AI Score
        ai_score = gr.Markdown("🤖 **AI**: 0/0 (100%)", elem_classes="score-box")
        # Human Score
        human_score = gr.Markdown("👤 **Human**: 0/0 (100%)", elem_classes="score-box")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Image display
            image_display = gr.Image(
                label="Can you tell if this face is real or AI-generated?",
                type="filepath",
                height=400
            )
            
            # Start button (visible initially)
            start_btn = gr.Button("🎮 Start New Game", variant="primary", size="lg", visible=True)
        
        with gr.Column(scale=1):
            # Chat/conversation area
            chat_display = gr.Markdown(
                "👤 **You**: _Waiting to start..._\n\n🤖 **AI**: _Ready when you are!_",
                elem_classes="chat-box"
            )
            
            # Action buttons
            with gr.Row():
                ai_btn = gr.Button("🤖 AI-Generated", variant="secondary", size="lg", visible=False)
                real_btn = gr.Button("👤 Real", variant="secondary", size="lg", visible=False)
            
            next_btn = gr.Button("➡️ Next Round", variant="primary", size="lg", visible=False)
    
    # Feedback area
    feedback = gr.Markdown("", elem_classes="feedback-box")
    
    gr.Markdown("""
    ---
    ### 📊 About This Game:
    This interactive game uses a deep learning model trained to distinguish between real human faces and AI-generated faces.
    The model has been trained on thousands of images and uses advanced computer vision techniques.
    
    **Can you beat the AI at its own game?** 🤔
    """)
    
    # Event handlers
    start_btn.click(
        fn=start_game,
        inputs=[],
        outputs=[image_display, chat_display, ai_score, human_score, ai_btn, real_btn, next_btn, feedback, start_btn]
    )
    
    ai_btn.click(
        fn=lambda: make_prediction("AI-Generated"),
        inputs=[],
        outputs=[chat_display, ai_score, human_score, ai_btn, real_btn, next_btn, feedback]
    )
    
    real_btn.click(
        fn=lambda: make_prediction("Real"),
        inputs=[],
        outputs=[chat_display, ai_score, human_score, ai_btn, real_btn, next_btn, feedback]
    )
    
    next_btn.click(
        fn=next_round,
        inputs=[],
        outputs=[image_display, chat_display, ai_btn, real_btn, next_btn, feedback, start_btn]
    )

# ============================================================
# Launch the app
# ============================================================

if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Allows external connections
        server_port=7860  # Default Gradio port
    )
