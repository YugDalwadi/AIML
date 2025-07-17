import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ENC_HIDDEN_SIZE = 1
DEC_HIDDEN_SIZE = 1
BATCH_SIZE = 64


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.BiRNN = nn.RNN(bidirectional=True,
                            input_size=1,
                            batch_first=True,
                            hidden_size=ENC_HIDDEN_SIZE)

    def forward(self, x):  # x is of shape (batch_size, seq_length, 1)
        out = self.BiRNN(x)
        return out[0]  # returns annotation vectors


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_size = ENC_HIDDEN_SIZE*2  # *2 due to biRnn
        self.decoder = nn.RNN(input_size=input_size,
                              hidden_size=DEC_HIDDEN_SIZE,
                              batch_first=True)

    def forward(self, context_vecs):
        hiddenStates, nextWord = self.decoder(context_vecs)
        return hiddenStates, nextWord


class AlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.alignDecoder = nn.Linear(DEC_HIDDEN_SIZE, 256)
        self.alignAnnotation = nn.Linear(2*ENC_HIDDEN_SIZE, 256)
        self.hidden = nn.Linear(256, 1)

    def forward(self, decoderState, annotation):

        alignedAnnotation: torch.Tensor = self.alignAnnotation(annotation)
        print(f"Size of aligned annotation: {alignedAnnotation.size()}")

        alignedDecoder: torch.Tensor = self.alignDecoder(decoderState)
        alignedDecoder_exp = alignedDecoder.unsqueeze(
            1).repeat(1, alignedAnnotation.size(1), 1)
        print(f"Size of aligned decoder: {alignedDecoder_exp.size()}")

        energies: torch.Tensor = self.hidden(
            F.tanh(alignedAnnotation + alignedDecoder_exp))

        return energies


class AttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.alignment = AlignmentModel()
        self.prevHiddenState = None

    def forward(self, x: torch.Tensor):
        print(f"Size of input is: {x.size()}")

        annotations: torch.Tensor = self.encoder(x)
        print(f"Size of annotations is: {annotations.size()}")

        if (self.prevHiddenState == None):
            prevHidden = torch.rand(
                size=(BATCH_SIZE, DEC_HIDDEN_SIZE)).to(device)
            print(f"size of dummy prevHidden: {prevHidden.size()}")

            energies = self.alignment(prevHidden, annotations)
        else:
            print(f"size of prevHidden: {self.prevHiddenState.size()}")
            energies = self.alignment(self.prevHiddenState, annotations)

        exp_energies = torch.exp(energies)
        weights = exp_energies/(torch.sum(exp_energies))
        print(f"Size of weights is: {weights.size()}")

        contextVector = torch.sum(torch.mul(annotations, weights), dim=1)
        # print(contextVector)
        print(f"Size of contextVectors: {contextVector.size()}")

        predSentences, hiddenState = self.decoder(contextVector)
        self.prevHiddenState = hiddenState
        print(f"Size of predicted sentences is: {predSentences.size()}")
