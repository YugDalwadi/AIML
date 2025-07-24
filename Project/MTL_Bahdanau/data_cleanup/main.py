import pandas as pd
import helpers
import clean_funcs

# df = pd.DataFrame(columns=["English", "French"])
# df = pd.read_csv(r"data_cleanup/data/fra-eng/fra.txt",
#                  sep='\t', encoding="utf-8")
# source, target = df["English"], df["French"]

# source_clean = clean_funcs.dataCleaner(source)
# target_clean = clean_funcs.dataCleaner(target)

# source_tok = clean_funcs.tokenizer(
#     source_clean, "eng")
# target_tok = clean_funcs.tokenizer(
#     target_clean, "fra")

# helpers.savePickle("source_tok", source_tok, "target_tok", target_tok)
source_tok, target_tok = helpers.loadPickle(
    "pickles/source_tok", "pickles/target_tok")
source_pad = clean_funcs.padData(source_tok)
target_pad = clean_funcs.padData(target_tok)

source_pad = clean_funcs.insertSpecialTokens(source_tok)
target_pad = clean_funcs.insertSpecialTokens(target_tok)
helpers.savePickle("pickles/source_pad", source_pad,
                   "pickles/target_pad", target_pad)

# source_pad, target_pad = helpers.loadPickle(
#     "pickles/source_pad", "pickles/target_pad")
# print(len(source_pad), len(target_pad))
source_trimmed, target_trimmed = clean_funcs.trimData(source_pad, target_pad)

helpers.savePickle("pickles/source_trim", source_trimmed,
                   "pickles/target_trim", target_trimmed)

# source_trimmed, target_trimmed = helpers.loadPickle(
#     "pickles/source_trim", "pickles/target_trim")
source_encodings, source_freq = clean_funcs.getWordEncAndFreq(source_trimmed)
target_encodings, target_freq = clean_funcs.getWordEncAndFreq(target_trimmed)

source_enc = clean_funcs.replaceUnk(source_trimmed, source_freq)
target_enc = clean_funcs.replaceUnk(target_trimmed, target_freq)
helpers.savePickle("pickles/source_enc", source_enc,
                   "pickles/target_enc", target_enc)
helpers.savePickle("pickles/eng_encodings", source_encodings,
                   "pickles/fra_encodings", target_encodings)
source_encodings, source_freq = clean_funcs.getWordEncAndFreq(source_trimmed)
target_encodings, target_freq = clean_funcs.getWordEncAndFreq(target_trimmed)

helpers.savePickle("pickles/source_freq", source_freq,
                   "pickles/target_freq", target_freq)

source_final = clean_funcs.encodeData(source_enc, source_encodings)
target_final = clean_funcs.encodeData(target_enc, target_encodings)

helpers.savePickle("pickles/source_final", source_final,
                   "pickles/target_final", target_final)
