# Natural Disaster Tweet Classification using BERT

## Approach
We tackled the challenge of identifying tweets about natural disasters using BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art model for NLP tasks such as text classification.

### Data
- **Source**: Kaggle
- **Training Data**: A CSV file with tweets, keywords, locations, and a binary identifier for natural disasters.
- **Testing Data**: A similar CSV file without the binary identifier.

### Preprocessing
- **Training**: Data split into text, keywords, and location subsets. Entries missing keywords or location were pruned.
- **Testing**: Data split similarly to training data.

### Model Training
Each subset (text, keywords, location) was used to fine-tune a pre-trained BERT model:
- **Layers**: Input (text/keyword/location), preprocessing (tokenization), BERT encoding, dropout (to prevent overfitting), dense (final score).
- **Activation**: Sigmoid function for binary classification.
- **Parameters**: Various epoch sizes, batch sizes, and randomness to optimize performance.
- **Storage**: Trained models saved for future use to avoid re-training.

## Implementation Details
- **Environment**: Google Colab with high-performance GPU runtime (40GB GRAM, 82.5GB RAM).
- **Packages**: TensorFlow, TensorFlow Hub, TensorFlow Text, AdamW optimizer.
- **Epochs**: Optimal values found through experimentation - text (5), keywords (7), location (3-4).
- **Batch Size**: 64 to balance memory usage and performance.

## Experiments
Three models trained:
- **Text Model**: 5 epochs.
- **Keywords Model**: 7 epochs.
- **Location Model**: 3-4 epochs.

## Discussion
- **Ranking**: Our model ranks 242nd among over 1000 submissions.
- **Performance**: Text model outperformed others in prediction accuracy.
- **Future Work**: Propose weighting models based on training accuracy for combined predictions. Suggest using TPUs for better performance due to optimized tensor operations.

## Competition Description
Twitter is crucial during emergencies. This competition aims to develop a model to predict which tweets are about real disasters. The dataset includes 10,000 hand-classified tweets. New to NLP? Start with our tutorial.

**Tweet source**: [Example Tweet](https://twitter.com/AnyOtherAnnaK/status/629195955506708480)

## Contributor
Nuo Xu  
Daniel Ye
