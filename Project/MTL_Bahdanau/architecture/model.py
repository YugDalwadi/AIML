import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ENC_HIDDEN_SIZE = 128
DEC_HIDDEN_SIZE = 128
MAX_OUTPUT_LEN = 16
UNIQUE_FRA_WORDS = 7576
EOS, SOS, PAD, UNK = 2, 1, 0, 3
EMB_DIM = 128
TOP_K = 5

# function for extracting tokens using multinomial sampling.


def getTokensFromLogits(nextTokenLogits) -> torch.Tensor:
    nextTokenProbs = F.softmax(nextTokenLogits, dim=-1)
    topkProbs, topkIndices = torch.topk(
        nextTokenProbs, k=TOP_K, dim=-1)

    topkProbs = topkProbs.squeeze(1)      # (B, TOP_K)
    topkIndices = topkIndices.squeeze(1)  # (B, TOP_K)

    sampled_k_indices = torch.multinomial(
        topkProbs, num_samples=1)  # (B, 1)
    predTokens = torch.gather(
        topkIndices, 1, sampled_k_indices).long()   # (B, 1)

    return predTokens


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.BiRNN = nn.LSTM(bidirectional=True,
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

        exp_energies = torch.exp(energies)

        # To match dimensions for division, we unsqueeze
        energies_sum = torch.sum(  # (B,1,1) --> 1 at end added due to unsqueeze
            exp_energies, dim=1).unsqueeze(-1)

        weights = exp_energies / energies_sum  # (B, SEQ_LEN, 1)
        # print(f"Size of weights is: {weights.size()}")
        return weights


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # *2 due to biRnn, *2 again due to concat with prev word
        input_size = ENC_HIDDEN_SIZE*2 + EMB_DIM

        self.decoder = nn.LSTM(input_size=input_size,
                               hidden_size=DEC_HIDDEN_SIZE,
                               batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.unembedder = nn.Linear(DEC_HIDDEN_SIZE, UNIQUE_FRA_WORDS)

    def forward(self, contextVecs, prevStates):
        adjPrevStates = [[], []]
        for i in range(len(prevStates)):  # adjust both hidden states and cell states
            adjPrevStates[i] = prevStates[i].squeeze(1).unsqueeze(0)

        _, predWordState = self.decoder(  # (1,B,DEC_HIDDEN_SIZE) is predWordState. This is standard output for the nn RNN
            contextVecs, adjPrevStates)

        adjPredWordStates = [[], []]
        for i in range(len(predWordState)):
            adjPredWordStates[i] = predWordState[i].squeeze(
                0).unsqueeze(1)  # (B,DEC_HIDDEN_SIZE)
        adjPredWordHiddenStates = self.dropout1(adjPredWordStates[0])
        # logits for predWord. Will be used for training
        predWordLogits = self.unembedder(adjPredWordHiddenStates)  # ()
        return predWordLogits, adjPredWordStates


class StartingHiddenState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2*ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE)

    def forward(self, averagedAnnotation):
        hiddenState = self.hidden(averagedAnnotation)

        return hiddenState


class StartingCellState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2*ENC_HIDDEN_SIZE, DEC_HIDDEN_SIZE)

    def forward(self, averagedAnnotation):
        cellState = self.hidden(averagedAnnotation)

        return cellState


class Embedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.embeddingLayer = nn.Embedding(
            UNIQUE_FRA_WORDS, EMB_DIM, padding_idx=PAD)

    def forward(self, token):
        embeds = self.embeddingLayer(token)
        embeds = self.dropout1(embeds)
        return embeds


class AttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.alignment = AlignmentModel()
        self.embedder = Embedder()
        self.initHidden = StartingHiddenState()
        self.initCell = StartingCellState()
        self.TFR = 0.75

    # input: (B,SEQ_LEN,1)--> means (Batch, (vector size) ). vector is (SEQ_LEN,1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # print(f"Size of input is: {x.size()}")

        annotations: torch.Tensor = self.encoder(
            x)  # (B,SEQ_LEN,ENC_HIDDEN_DIM)
        batch_size = x.size(0)
        sos = torch.full((batch_size, 1,), SOS,
                         dtype=torch.long).to(device)  # (B,1)

        prevHiddenState = prevCellState = None
        prevWordEmbedding = self.embedder(sos)  # (B,1,128)
        predSentences = [[] for _ in range(batch_size)]
        logitsStored = []

        for ix in range(MAX_OUTPUT_LEN):

            if (prevHiddenState == None):  # For a batch's first iter.
                avgAnnotation = torch.sum(
                    annotations, dim=1, keepdim=True)/annotations.size(dim=1)
                prevHiddenState = self.initHidden(avgAnnotation)

            if (prevCellState == None):
                avgAnnotation = torch.sum(
                    annotations, dim=1, keepdim=True)/annotations.size(dim=1)
                prevCellState = self.initCell(avgAnnotation)

            weights = self.alignment(  # (B,SEQ_LEN)
                prevHiddenState, annotations)

            # context vectors for current word pred for entire batch
            contextVector = torch.sum(  # (B, 1, ENC_HIDDEN_DIM)
                torch.mul(annotations, weights), dim=1, keepdim=True)
            # print(f"Size of contextVectors: {contextVector.size()}")

            tfrNo = torch.rand(1).item()
            if (tfrNo <= self.TFR):  # Case where teacher forcing will be applied
                trueWord = y[:, ix]
                trueWordEmbeds = self.embedder(trueWord)

                concatDecoderInput = torch.concat(  # (B,1, ENC_HIDDEN_DIM + EMBEDDING_DIM)
                    (contextVector, trueWordEmbeds), dim=-1)

            else:
                concatDecoderInput = torch.concat(
                    (contextVector, prevWordEmbedding), dim=-1)

            nextTokenLogits, currStates = self.decoder(  # [ (B, 1, UNIQUE_FRA_WORDS), (B, 1, DEC_HIDDEN_SIZE) ]
                concatDecoderInput, (prevHiddenState, prevCellState))
            prevHiddenState = currStates[0]
            prevCellState = currStates[1]

            # predTokens = getTokensFromLogits(nextTokenLogits)
            predTokens = torch.argmax(nextTokenLogits, dim=-1)

            currWordEmbedding = self.embedder(predTokens)
            prevWordEmbedding = currWordEmbedding
            logitsStored.append(nextTokenLogits)

        logitsStored = torch.cat(logitsStored, dim=1)

        return logitsStored

    def predict(self, x):

        annotations: torch.Tensor = self.encoder(x)
        batch_size = x.size(0)
        sos = torch.full((batch_size, 1,), SOS,
                         dtype=torch.long).to(device)
        prevWordEmbedding = self.embedder(sos)
        prevHiddenState = None
        prevCellState = None
        predSentences = [[] for _ in range(batch_size)]

        for _ in range(MAX_OUTPUT_LEN):

            if (prevHiddenState == None):  # For a batch's first iter.
                avgAnnotation = torch.sum(
                    annotations, dim=1, keepdim=True)/annotations.size(dim=1)
                prevHiddenState = self.initHidden(avgAnnotation)

            if (prevCellState == None):
                avgAnnotation = torch.sum(
                    annotations, dim=1, keepdim=True)/annotations.size(dim=1)
                prevCellState = self.initCell(avgAnnotation)

            weights = self.alignment(
                prevHiddenState, annotations)

            contextVector = torch.sum(
                torch.mul(annotations, weights), dim=1, keepdim=True)

            concatDecoderInput = torch.concat(
                (contextVector, prevWordEmbedding), dim=-1)

            nextTokenLogits, currStates = self.decoder(
                concatDecoderInput, (prevHiddenState, prevCellState))
            prevHiddenState = currStates[0]
            prevCellState = currStates[1]

            predTokens = getTokensFromLogits(nextTokenLogits)

            currWordEmbedding = self.embedder(predTokens)
            prevWordEmbedding = currWordEmbedding

            for i in range(len(predSentences)):
                if predSentences[i]:
                    if (predSentences[i][-1] != EOS):
                        predSentences[i].append(predTokens[i].item())
                else:
                    predSentences[i].append(predTokens[i].item())

        return predSentences
