class JTC_config_train():
    
    BERT_MODEL = "google-bert/bert-base-multilingual-uncased"
    MAX_SEQ_LENGTH=100
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 20
    LEARNING_RATE = 5e-5
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 5
    OUTPUT_DIR = "/Users/user/Desktop/jarvis/Jarvis/"
    MODEL_FILE_NAME = "textclass{epoch}.pt"
    PATIENCE = 2
    