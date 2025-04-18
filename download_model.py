from transformers import AutoModel, AutoTokenizer
model_name = "TalTechNLP/whisper-large-v3-turbo-et-subs"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("./taltech_model")
tokenizer.save_pretrained("./taltech_model")