"""Gradio UI for Mood Axis."""

import gradio as gr
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import sys

sys.path.insert(0, str(__file__).rsplit("/src/", 1)[0])
from config.settings import MOOD_AXES, AXIS_LABELS, DEFAULT_MODEL, MODEL_BASELINE
from src.model.loader import ModelManager
from src.model.inference import generate_with_hidden_states, format_chat_messages
from src.mood.projector import MoodProjector, MoodReading
from src.ui.charts import create_spider_chart, get_mood_glow_css, create_metrics_html


# Global state
_projector: Optional[MoodProjector] = None
_model_name: str = DEFAULT_MODEL
_session_history: List[Dict] = []
_previous_mood: Optional[Dict[str, float]] = None


def get_projector() -> MoodProjector:
    """Get or create the mood projector."""
    global _projector
    if _projector is None:
        _projector = MoodProjector()
    return _projector


def extract_text_content(content) -> str:
    """Extract plain text from various content formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        if "text" in content:
            return extract_text_content(content["text"])
        elif "content" in content:
            return extract_text_content(content["content"])
        return str(content)
    elif isinstance(content, list) and len(content) > 0:
        return extract_text_content(content[0])
    return str(content)


def create_mood_plot(
    reading: Optional[MoodReading] = None,
    deltas: Optional[Dict[str, float]] = None,
    previous_values: Optional[Dict[str, float]] = None,
    use_baseline: bool = False,
) -> go.Figure:
    """Create a spider chart for mood visualization.

    Args:
        reading: Optional mood reading to display
        deltas: Optional dict of value changes from previous message
        previous_values: Optional previous mood values for ghost trail
        use_baseline: If True and no reading, show model baseline instead of zeros

    Returns:
        Plotly figure with spider chart
    """
    if reading is None:
        # Show model baseline values at startup
        values = MODEL_BASELINE.copy() if use_baseline else {
            "warm_cold": 0, "patient_irritated": 0,
            "confident_cautious": 0, "proactive_reluctant": 0
        }
    else:
        values = reading.values

    return create_spider_chart(
        values=values,
        previous_values=previous_values,
        show_ghost=previous_values is not None,
        animated=True,
    )


def chat_response(
    message: str,
    history: List[List[str]],
    system_prompt: str,
) -> Tuple[str, go.Figure, str]:
    """Generate a response and compute mood.

    Args:
        message: User message
        history: Chat history
        system_prompt: System prompt

    Returns:
        Tuple of (response_text, mood_plot, mood_summary)
    """
    global _session_history

    # Get model
    model, tokenizer = ModelManager.get_model(_model_name)
    projector = get_projector()

    # Format history for the model (Gradio 6 uses messages format)
    chat_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            content = extract_text_content(msg["content"])
            chat_history.append({"role": msg["role"], "content": content})
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            chat_history.append({"role": "user", "content": extract_text_content(msg[0])})
            if msg[1]:
                chat_history.append({"role": "assistant", "content": extract_text_content(msg[1])})

    # Create messages
    messages = format_chat_messages(
        user_message=message,
        system_message=system_prompt if system_prompt.strip() else None,
        history=chat_history,
    )

    # Generate with hidden states
    result = generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
    )

    global _previous_mood

    # Project to mood axes
    reading = projector.project(result.hidden_state)

    # Calculate deltas from previous message
    deltas = {}
    if _previous_mood:
        for axis in MOOD_AXES:
            current = reading.values.get(axis, 0)
            previous = _previous_mood.get(axis, 0)
            deltas[axis] = current - previous

    # Create visualization with ghost trail
    fig = create_mood_plot(
        reading,
        deltas if _previous_mood else None,
        previous_values=_previous_mood,
    )

    # Build summary with deltas
    summary_parts = []
    for axis in MOOD_AXES:
        pos_label, neg_label = AXIS_LABELS.get(axis, ("Pos", "Neg"))
        value = reading.values.get(axis, 0)
        value_str = f"{value:+.2f}"

        if deltas and axis in deltas:
            delta = deltas[axis]
            delta_str = f"({delta:+.2f})" if delta != 0 else "(=)"
            summary_parts.append(f"{pos_label}: {value_str} {delta_str}")
        else:
            summary_parts.append(f"{pos_label}: {value_str}")

    summary = "\n".join(summary_parts)

    # Save to session history
    _session_history.append({
        "timestamp": datetime.now().isoformat(),
        "user_message": message,
        "assistant_response": result.text,
        "mood_values": reading.values,
        "mood_deltas": deltas,
        "mood_descriptions": reading.descriptions,
    })

    # Update previous mood for next comparison
    _previous_mood = reading.values.copy()

    return result.text, fig, summary


def export_session() -> str:
    """Export session history as JSON.

    Returns:
        JSON string of session history
    """
    return json.dumps(_session_history, indent=2, ensure_ascii=False)


def clear_session():
    """Clear session history."""
    global _session_history, _previous_mood
    _session_history = []
    _previous_mood = None
    return [], create_mood_plot(None), "Session cleared"


def create_app(model_name: str = DEFAULT_MODEL) -> gr.Blocks:
    """Create the Gradio application.

    Args:
        model_name: Model to use for generation

    Returns:
        Gradio Blocks application
    """
    global _model_name
    _model_name = model_name

    with gr.Blocks(title="Mood Axis") as app:
        # Inject glow CSS
        gr.HTML(get_mood_glow_css())

        gr.Markdown("# 🎭 Mood Axis")
        gr.Markdown("*Visualize the 'mood' of LLM responses across three axes*")

        with gr.Row():
            # Left column: Chat
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=450,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type a message...",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                system_prompt = gr.Textbox(
                    label="System Prompt (optional)",
                    placeholder="Enter a system prompt to influence the model's behavior...",
                    lines=2,
                )

            # Right column: Mood indicators
            with gr.Column(scale=2):
                gr.Markdown("### 🕸️ Mood Spider")
                gr.Markdown("*Model baseline (typical neutral response)*", elem_id="baseline-hint")
                with gr.Group():
                    gr.HTML('<div class="mood-spider-container"><div class="mood-spider-inner">')
                    mood_plot = gr.Plot(
                        value=create_mood_plot(None, use_baseline=True),
                        label="",
                        show_label=False,
                    )
                    gr.HTML('</div></div>')

                mood_metrics = gr.HTML(
                    value=create_metrics_html(MODEL_BASELINE),
                    label="",
                )
                mood_summary = gr.Textbox(
                    label="Mood Summary",
                    interactive=False,
                    lines=3,
                    visible=False,  # Hidden - metrics HTML shows this now
                )

                with gr.Row():
                    clear_btn = gr.Button("Clear Session")
                    export_btn = gr.Button("Export Session")

                export_output = gr.Textbox(
                    label="Exported JSON",
                    interactive=False,
                    lines=5,
                    visible=False,
                )

        # Event handlers
        def respond(message, history, system_prompt):
            response, fig, summary = chat_response(message, history, system_prompt)
            # Gradio 6 messages format
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]

            # Get current mood values and deltas for metrics HTML
            if _session_history:
                last = _session_history[-1]
                values = last.get("mood_values", {})
                deltas = last.get("mood_deltas", {})
                metrics_html = create_metrics_html(values, deltas)
            else:
                metrics_html = create_metrics_html(
                    {"warm_cold": 0, "patient_irritated": 0, "confident_cautious": 0}
                )

            return "", history, fig, summary, metrics_html

        msg.submit(
            respond,
            [msg, chatbot, system_prompt],
            [msg, chatbot, mood_plot, mood_summary, mood_metrics],
        )

        send_btn.click(
            respond,
            [msg, chatbot, system_prompt],
            [msg, chatbot, mood_plot, mood_summary, mood_metrics],
        )

        def do_clear_session():
            history, fig, summary = clear_session()
            # Return to baseline on clear
            fig = create_mood_plot(None, use_baseline=True)
            metrics_html = create_metrics_html(MODEL_BASELINE)
            return history, fig, summary, metrics_html

        clear_btn.click(
            do_clear_session,
            [],
            [chatbot, mood_plot, mood_summary, mood_metrics],
        )

        def show_export():
            return gr.update(visible=True, value=export_session())

        export_btn.click(
            show_export,
            [],
            [export_output],
        )

        # Instructions
        gr.Markdown("""
        ### How to Use
        1. Type a message and press Enter or click Send
        2. Watch the mood indicators update with each response
        3. Try different system prompts to influence the model's mood
        4. Export your session to save the conversation and mood data

        ### Test Scenarios
        - **Neutral**: Ask factual questions → indicators should be near 0
        - **Warm**: "Please be more friendly" → Warm axis should increase
        - **Confident**: "Be more confident in your answers" → Confident axis should increase
        """)

    return app


def launch_app(
    model_name: str = DEFAULT_MODEL,
    share: bool = False,
    server_port: int = 7860,
):
    """Launch the Gradio application.

    Args:
        model_name: Model to use
        share: Whether to create a public link
        server_port: Port to run on
    """
    # Pre-load model
    print(f"Loading model {model_name}...")
    ModelManager.get_model(model_name)
    print("Model loaded!")

    # Pre-load projector (will fail gracefully if not calibrated)
    try:
        get_projector()
        print("Mood projector loaded!")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Mood indicators will not work until calibration is complete.")

    app = create_app(model_name)
    app.launch(
        share=share,
        server_port=server_port,
    )


if __name__ == "__main__":
    launch_app()
