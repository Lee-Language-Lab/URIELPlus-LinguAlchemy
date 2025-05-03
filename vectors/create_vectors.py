# Create vectors for languages in the MASSIVE dataset

# Import URIEL+ for retrieving language vectors
from urielplus import urielplus as uriel
import torch
import numpy as np

# Initialize the URIEL+ system and enable caching for efficiency
u = uriel.URIELPlus()
u.reset()
u.set_cache(True)
u.integrate_databases()        # Integrate all linguistic sources
u.softimpute_imputation()      # Fill missing values using soft imputation

# Load existing MASSIVE dataset
massive_dataset = torch.load('vectors\\massive_vectors\\URIEL\\syntax_knn.pt', weights_only=False)

# Languages in the MASSIVE dataset
MASSIVE_LANGS = {
    "af-ZA": "afri1274",
    "am-ET": "amha1245",
    "ar-SA": "stan1318",
    "az-AZ": "nort2697",
    "bn-BD": "beng1280",
    "ca-ES": "stan1289",
    "cy-GB": "wels1247",
    "da-DK": "dani1285",
    "de-DE": "stan1295",
    "el-GR": "mode1248",
    "en-US": "stan1293",
    "fa-IR": "west2369",
    "fi-FI": "finn1318",
    "fr-FR": "stan1290",
    "he-IL": "hebr1245",
    "hi-IN": "hind1269",
    "hu-HU": "hung1274",
    "hy-AM": "nucl1235",
    "id-ID": "indo1316",
    "is-IS": "icel1247",
    "it-IT": "ital1282",
    "jv-ID": "java1254",
    "ja-JP": "nucl1643",
    "kn-IN": "nucl1305",
    "ka-GE": "nucl1302",
    "km-KH": "cent1989",
    "ko-KR": "kore1280",
    "lv-LV": "latv1249",
    "ml-IN": "mala1464",
    "mn-MN": "mong1331",
    "ms-MY": "stan1306",
    "my-MM": "nucl1310",
    "nl-NL": "dutc1256",
    "nb-NO": "norw1259",
    "pl-PL": "poli1260",
    "pt-PT": "port1283",
    "ro-RO": "roma1327",
    "ru-RU": "russ1263",
    "sl-SL": "slov1268",
    "es-ES": "stan1288",
    "sq-AL": "gheg1238",
    "sw-KE": "swah1253",
    "sv-SE": "swed1254",
    "ta-IN": "tami1289",
    "te-IN": "telu1262",
    "tl-PH": "taga1270",
    "th-TH": "thai1261",
    "tr-TR": "nucl1301",
    "ur-PK": "urdu1245",
    "vi-VN": "viet1252",
    "zh-TW": "mand1415",
    "zh-CN": "mand1415"
}

# Retrieve syntactic vectors from URIEL+
massive_vectors = u.get_vector("syntactic", list(MASSIVE_LANGS.values()))

# Update MASSIVE dataset entries with URIEL+ vectors as NumPy arrays
for lang in massive_dataset:
    if lang in MASSIVE_LANGS:
        uriel_code = MASSIVE_LANGS[lang]
        if uriel_code in massive_vectors:
            value = massive_vectors[uriel_code]
            
            # If the value is a list, convert it to a numpy array
            if isinstance(value, list):
                value = np.array(value)
            
            # Save the NumPy array to massive_dataset
            massive_dataset[lang] = value
        else:
            print(f"Missing vector for {uriel_code}")

# After converting all lists to numpy arrays, save the dataset
torch.save(massive_dataset, "syntax_knn.pt")


# # Languages in the MasakhaNews dataset
# MASAKHANEWS_LANGS = [
#     "amha1245", "stan1293", "stan1290", "haus1257", "nucl1417", "ling1263",
#     "gand1255", "east2652", "nige1257", "rund1242", "shon1251", "soma1255",
#     "swah1253", "tigr1271", "xhos1239", "yoru1245"
# ]