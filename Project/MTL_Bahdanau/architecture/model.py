from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ENC_HIDDEN_SIZE = 32
DEC_HIDDEN_SIZE = 16
MAX_OUTPUT_LEN = 11
UNIQUE_FRA_WORDS = 6754
EOS, SOS, PAD, UNK = 2, 1, 0, 3
EMB_DIM = 128


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


class AlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.alignDecoder = nn.Linear(DEC_HIDDEN_SIZE, 256)
        self.alignAnnotation = nn.Linear(2*ENC_HIDDEN_SIZE, 256)
        self.hidden = nn.Linear(256, 1)

    def forward(self, decoderState, annotation):

        alignedAnnotation: torch.Tensor = self.alignAnnotation(annotation)
        # print(f"Size of aligned annotation: {alignedAnnotation.size()}")

        alignedDecoder: torch.Tensor = self.alignDecoder(decoderState)

        # print(f"Size of aligned decoder: {alignedDecoder.size()}")

        energies: torch.Tensor = self.hidden(
            F.tanh(alignedAnnotation + alignedDecoder))

        return energies


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # *2 due to biRnn, *2 again due to concat with prev word
        input_size = ENC_HIDDEN_SIZE*2 + EMB_DIM
        self.decoder = nn.RNN(input_size=input_size,
                              hidden_size=DEC_HIDDEN_SIZE,
                              batch_first=True)
        self.unembedder = nn.Linear(DEC_HIDDEN_SIZE, 6754)

    def forward(self, contextVecs, prevHiddenState):

        _, predWordState = self.decoder(  # (1,B,DEC_HIDDEN_SIZE) is predWordState. This is standard output for the nn RNN
            contextVecs, prevHiddenState.squeeze(1).unsqueeze(0))
        predWordState = predWordState.squeeze(0)  # (B,DEC_HIDDEN_SIZE)
        predWordState = predWordState.unsqueeze(1)  # (B,1,DEC_HIDDEN_SIZE)

        # logits for predWord. Will be used for training
        predWordLogits = self.unembedder(predWordState)  # ()
        return predWordLogits, predWordState


class StartingHiddenState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2*ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE)

    def forward(self, averagedAnnotation):
        hiddenState = self.hidden(averagedAnnotation)

        return hiddenState


class Embedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddingLayer = nn.Embedding(UNIQUE_FRA_WORDS, EMB_DIM)

    def forward(self, token):
        embeds = self.embeddingLayer(token)
        return embeds


class AttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.alignment = AlignmentModel()
        self.embedder = Embedder()
        self.initHidden = StartingHiddenState()

    # input: (B,SEQ_LEN,1)--> means (Batch, (vector size) ). vector is (SEQ_LEN,1)
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # print(f"Size of input is: {x.size()}")

        annotations: torch.Tensor = self.encoder(
            x)  # (B,SEQ_LEN,ENC_HIDDEN_DIM)
        # print(f"Size of annotations is: {annotations.size()}")
        batch_size = x.size(0)
        sos = torch.full((batch_size, 1,), SOS,
                         dtype=torch.long).to(device)  # (B,1)

        prevHiddenState = None
        prevWordEmbedding = self.embedder(sos)  # (B,1,128)
        predSentences = [[] for _ in range(batch_size)]
        logitsStored = []

        for ix in range(MAX_OUTPUT_LEN):

            if (prevHiddenState == None):  # For a batch's first iter.
                avgAnnotation = torch.sum(
                    annotations, dim=1, keepdim=True)/annotations.size(dim=1)

                prevHiddenState = self.initHidden(avgAnnotation)
                # print(f"size of dummy prevHidden: {prevHiddenState.size()}")

                energies = self.alignment(  # (B,SEQ_LEN,1)
                    prevHiddenState, annotations)

            else:  # For subsequent iters of same batch

                # print(f"size of prevHidden: {prevHiddenState.size()}")
                energies = self.alignment(  # (B,SEQ_LEN)
                    prevHiddenState, annotations)

            exp_energies = torch.exp(energies)

            # To match dimensions for division, we unsqueeze
            energies_sum = torch.sum(  # (B,1,1) 1 at end added due to unsqueeze
                exp_energies, dim=1).unsqueeze(-1)

            weights = exp_energies / energies_sum  # (B, SEQ_LEN, 1)
            # print(f"Size of weights is: {weights.size()}")

            # context vectors for current word pred for entire batch
            contextVector = torch.sum(  # (B, 1, ENC_HIDDEN_DIM)
                torch.mul(annotations, weights), dim=1, keepdim=True)
            # print(
            # f"Size of contextVectors: {contextVector.size()}, prevWordEmbed: {prevWordEmbedding.size()}")

            concatDecoderInput = torch.concat(  # (B,1, ENC_HIDDEN_DIM + EMBEDDING_DIM)
                (contextVector, prevWordEmbedding), dim=-1)
            # print(
            #     f"Size of concatInput: {concatDecoderInput.size()}, prevHiddenState: {prevHiddenState.size()}")

            teacherTokens = y[:, ix].unsqueeze(-1)
            print(teacherTokens.size(), prevHiddenState.size())
            nextTokenLogits, currHiddenState = self.decoder(  # [ (B, 1, UNIQUE_FRA_WORDS), (B, 1, DEC_HIDDEN_SIZE) ]
                concatDecoderInput, teacherTokens)

            predTokens = torch.argmax(nextTokenLogits, dim=2)  # (B,1)
            # print(f"Size of predTokens: {predTokens.size()}")

            # This keeps appending to predSentences as long as some sentence hasnt encountered an EOS. Then, that predSentence
            # will stop being written to.
            # for i in range(len(predSentences)):
            #     if predSentences[i]:
            #         if (predSentences[i][-1] != EOS and predTokens[i] != EOS):
            #             predSentences[i].append(predTokens[i].item())
            #     else:
            #         predSentences[i].append(predTokens[i].item())

            prevHiddenState = currHiddenState
            prevWordEmbedding = self.embedder(predTokens)
            logitsStored.append(nextTokenLogits)

        logitsStored = torch.cat(logitsStored, dim=1)
        # print(f"Shape of logitsStored: {logitsStored.size()}")

        return logitsStored, predSentences
