# Recommended Datasets for QLLM Dialogue Training

Here are excellent datasets for training your dialogue-based QLLM model with continuous learning capabilities:

## General Conversation Datasets

1. **DailyDialog** - A high-quality multi-turn dialogue dataset containing 13,118 conversations about daily life.
   - Size: ~13K conversations
   - Features: Clean, manually labeled emotions and dialogue acts
   - Already integrated in our code as `daily_dialog`

2. **PersonaChat/ConvAI2** - Conversations where each speaker has a defined persona.
   - Size: 10,907 dialogues
   - Features: Character personas, engaging conversations
   - Available as `conv_ai_2` in our system

3. **Empathetic Dialogues** - Conversations with emotional contexts, excellent for empathetic responses.
   - Size: 25K conversations
   - Features: Emotional grounding, empathetic responses
   - Integrated as `empathetic_dialogues`

## Knowledge-Grounded Conversations

4. **Wizard of Wikipedia** - Knowledge-focused conversations where one participant has access to Wikipedia.
   - Size: ~22K dialogues
   - Features: Knowledge grounding, factual consistency
   - Great for leveraging the memory extension

5. **TopicalChat** - Conversations grounded in news articles and fun facts.
   - Size: ~11K conversations
   - Features: Diverse topics, knowledge references

## Instruction and Feedback Datasets (Best for Continuous Learning)

6. **Anthropic Helpful and Harmless** (HH-RLHF) - High-quality preferred/dispreferred response pairs.
   - Size: 161K preference pairs
   - Features: Human feedback, perfect for continuous learning

7. **OpenAssistant Conversations** - Open-source assistant conversations with rankings and feedback.
   - Size: 66K conversations with human feedback
   - Features: Multi-lingual, diverse tasks, ranked responses

8. **Stanford Human Preferences (SHP)** - Human preference data on model responses.
   - Size: 385K samples with preferences
   - Features: High-quality preference data for continuous improvement

## Multi-domain and Task-oriented

9. **MultiWOZ** - Multi-domain wizard-of-oz conversations for task-oriented dialogue.
   - Size: 10K dialogues spanning multiple domains
   - Features: Task completion, domain transitions
   - Great for practical applications

## Implementation Strategy

For the best training approach with our framework:

```bash
# Start with baseline conversational ability
python train_dialogue.py --enable_extensions --dataset_name daily_dialog

# Then enhance with knowledge-grounded conversations
python train_dialogue.py --resume_from runs/dialogue_model/best_model.pt --enable_memory --dataset_name wizardofwikipedia

# Add continuous learning with feedback data
python train_dialogue.py --resume_from runs/dialogue_model/best_model.pt --resume_continuous_learning --continuous_learning --dataset_name OpenAssistant/oasst1

# Finally incorporate your custom data
python train_dialogue.py --resume_from runs/dialogue_model/best_model.pt --resume_continuous_learning --continuous_learning --data_path your_custom_feedback.json
```

This progressive approach fully leverages all three extensions:
- **Memory Extension**: Retains knowledge from previous conversations and datasets
- **Multimodal Extension**: Can be enhanced later with image-dialogue datasets
- **Quantum Extension**: Improves pattern recognition across all dialogue contexts

