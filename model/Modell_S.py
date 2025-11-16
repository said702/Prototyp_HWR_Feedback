import os
import torch
import torch.backends.cudnn as cudnn
from model.utils import CTCLabelConverter, AttnLabelConverter
from model.dataset import AlignCollate
from model.model import Model
import time

# Globale Konfiguration

Backend = torch.cuda.is_available()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, evaluation_loader, converter, opt):
    """Funktion für Vorhersagen ohne Accuracy-Berechnung."""
    model.eval()  # Modell in Evaluierungsmodus setzen
    predictions = []  # Liste zur Speicherung der Vorhersagen

    with torch.no_grad():  # Deaktivierung von Gradientenberechnung
        for image_tensors, _ in evaluation_loader:
            image_tensors = image_tensors.to(device)
            batch_size = image_tensors.size(0)

            # Initialisierung von Eingaben für die Decodierung
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            start_time = time.time()
            # Vorhersagen vom Modell
            preds = model(image_tensors, text_for_pred, is_train=False)

            # Maximal wahrscheinliche Zeichen extrahieren
            _, preds_index = preds.max(2)

            # Decodierung der Vorhersagen in lesbaren Text
            preds_str = converter.decode(preds_index, length_for_pred)
            elapsed_time = (time.time() - start_time) * 1000
            print(f"Vorhersagezeit: {elapsed_time:.2f} ms")


            # Entfernen von '[s]' Tokens (EOS)
            cleaned_preds = []
            for pred in preds_str:
                eos_index = pred.find('[s]')
                if eos_index != -1:
                    pred = pred[:eos_index]  # Alles nach '[s]' abschneiden
                cleaned_preds.append(pred.strip())  # Whitespace entfernen

            predictions.extend(cleaned_preds)  # Ergebnisse speichern

    return predictions

class Config:
    """Parameter für die Konfiguration."""
    def __init__(self):
        self.eval_data = "lmdb_dataset"
        self.benchmark_all_eval = False
        self.workers = 4
        self.batch_size = 192
        self.saved_model = "Extraction_Model/AttentionHTR.pth"
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.sensitive = True
        self.PAD = False
        self.data_filtering_off = False
        self.baiduCTC = False
        self.Transformation = "TPS"
        self.FeatureExtraction = "ResNet"
        self.SequenceModeling = "BiLSTM"
        self.Prediction = "Attn"
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.num_gpu = 0  # GPU-Zahl, Standard ist 0


def initialize_model():
    """Initialisiere das Modell und den Konverter."""
    opt = Config()

    # Konverter basierend auf Prediction-Typ
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    # Modell erstellen
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # Criterion basierend auf Prediction-Typ
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    # AlignCollate für Datenvorbereitung
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    return model, converter, criterion, align_collate, opt

def main():
    model, converter, criterion, AlignCollate_evaluation, opt = initialize_model()
    eval_data, eval_data_log = hierarchical_dataset(root="lmdb_dataset", opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
    eval_data, batch_size=opt.batch_size,shuffle=False,num_workers=int(0),collate_fn=AlignCollate_evaluation, pin_memory=True)
    predictions = predict(model, evaluation_loader, converter, opt)
    print("Vorhersagen abgeschlossen!")
    for i, pred in enumerate(predictions):
      print(f"Bild {i + 1}: {pred}")
   


if __name__ == "__main__":
    main()
