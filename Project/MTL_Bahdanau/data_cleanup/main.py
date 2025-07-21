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

# source_pad = clean_funcs.padData(source_tok)
# target_pad = clean_funcs.padData(target_tok)

# source_pad = clean_funcs.insertSpecialTokens(source_tok)
# target_pad = clean_funcs.insertSpecialTokens(target_tok)

# helpers.savePickle("source_pad", source_pad, "target_pad", target_pad)

# source_trimmed, target_trimmed = clean_funcs.trimData(source_pad, target_pad)

# helpers.savePickle("source_trim", source_trimmed,
#                    "target_trim", target_trimmed)

source_trimmed, target_trimmed = helpers.loadPickle(
    "pickles/source_trim", "pickles/target_trim")
print(len(source_trimmed), len(target_trimmed))
source_encodings, source_freq = clean_funcs.getWordEncAndFreq(source_trimmed)
target_encodings, target_freq = clean_funcs.getWordEncAndFreq(target_trimmed)

source_enc = clean_funcs.replaceUnk(source_trimmed, source_freq)
target_enc = clean_funcs.replaceUnk(target_trimmed, target_freq)
helpers.savePickle("source_enc", source_enc, "target_enc", target_enc)

source_encodings, source_freq = clean_funcs.getWordEncAndFreq(source_trimmed)
target_encodings, target_freq = clean_funcs.getWordEncAndFreq(target_trimmed)

helpers.savePickle("source_freq", source_freq, "target_enc", target_freq)

source_final = clean_funcs.encodeData(source_enc, source_encodings)
target_final = clean_funcs.encodeData(target_enc, target_encodings)

helpers.savePickle("source_final", source_final, "target_final", target_final)
