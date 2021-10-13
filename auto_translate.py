import streamlit as st
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


@st.cache
def load_data():
  supported_languages = [
    'ar_AR',
    'cs_CZ',
    'de_DE',
    'en_XX',
    'es_XX',
    'et_EE',
    'fi_FI',
    'fr_XX',
    'gu_IN',
    'hi_IN',
    'it_IT',
    'ja_XX',
    'kk_KZ',
    'ko_KR',
    'lt_LT',
    'lv_LV',
    'my_MM',
    'ne_NP',
    'nl_XX',
    'ro_RO',
    'ru_RU',
    'si_LK',
    'tr_TR',
    'vi_VN',
    'zh_CN',
    'af_ZA',
    'az_AZ',
    'bn_IN',
    'fa_IR',
    'he_IL',
    'hr_HR',
    'id_ID',
    'ka_GE',
    'km_KH',
    'mk_MK',
    'ml_IN',
    'mn_MN',
    'mr_IN',
    'pl_PL',
    'ps_AF',
    'pt_XX',
    'sv_SE',
    'sw_KE',
    'ta_IN',
    'te_IN',
    'th_TH',
    'tl_XX',
    'uk_UA',
    'ur_PK',
    'xh_ZA',
    'gl_ES',
    'sl_SI'
  ]
  return {k.split('_')[0]:k for k in supported_languages}

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    return (model, tokenizer)

data = load_data()

def translate_to_english(model, tokenizer, text):
    src_lang = detect(text)
    if src_lang in data:
        tokenizer.src_lang = src_lang
        encoded_txt = tokenizer(text, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_txt,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    else:
        print(f"Language {src_lang} not found")
        return

st.title("Auto Translate (To English)")


text = st.text_input(f"Write in any (1 of {len(data.keys())}) language")

st.text("What you wrote: ")

st.write(text)

st.text("English Translation: ")

if text:
    model, tokenizer = load_model()
    translated_text = translate_to_english(model, tokenizer, text)
    st.write(translated_text[0] if translated_text else "No translation found")
