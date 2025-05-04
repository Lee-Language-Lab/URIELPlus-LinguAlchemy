# Create vectors for languages in the MASAKHANEWS, MASSIVE, SEMREL datasets

import os
import torch
import numpy as np
from urielplus import urielplus as uriel

# Vector directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "experiment_vectors", "URIEL"))

# Function to update LinguAlchemy datasets
def update_dataset_with_vectors(dataset_filename, vector_type, experiment_dir, output_dir, languages):
    dataset_path = os.path.join(experiment_dir, dataset_filename)
    dataset = torch.load(dataset_path, weights_only=False)

    vectors = u.get_vector(vector_type, list(languages.values()))

    for lang_code, uriel_code in languages.items():
        if lang_code in dataset and uriel_code in vectors:
            vector = vectors[uriel_code]
            dataset[lang_code] = np.array(vector) if isinstance(vector, list) else vector
        elif lang_code in dataset:
            print(f"Missing vector for {uriel_code}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, dataset_filename)
    torch.save(dataset, output_path)

def create_concatenated_vector(first_dataset_filename, second_dataset_filename, experiment_dir, output_dir, languages):
    first_dataset_path = os.path.join(experiment_dir, first_dataset_filename)
    first_dataset = torch.load(first_dataset_path, weights_only=False)

    second_dataset_path = os.path.join(experiment_dir, second_dataset_filename)
    second_dataset = torch.load(second_dataset_path, weights_only=False)

    concatenated_vectors = {}

    for key in languages.keys():
        concatenated_vectors[key] = np.concatenate([first_dataset[key], second_dataset[key]], axis=0)

    output_filename = os.path.splitext(first_dataset_filename)[0] + '_' + os.path.splitext(second_dataset_filename)[0] + '.pt'
    output_path = os.path.join(output_dir, output_filename)

    torch.save(concatenated_vectors, output_path)

# Dictionary mapping the locale codes (language-country) from MASAKHANEWS languages to their corresponding glottocodes
MASAKHANEWS_LANGS = {
    "amh": "amha1245", "eng": "stan1293", "fra": "stan1290", "hau": "haus1257",
    "ibo": "nucl1417", "lin": "ling1263", "lug": "gand1255", "orm": "east2652",
    "pcm": "nige1257", "run": "rund1242", "sna": "shon1251", "som": "soma1255",
    "swa": "swah1253", "tir": "tigr1271", "xho": "xhos1239", "yor": "yoru1245"
}

# Dictionary mapping the locale codes (language-country) from MASSIVE languages to their corresponding glottocodes
MASSIVE_LANGS = {
    "af-ZA": "afri1274", "am-ET": "amha1245", "ar-SA": "stan1318", "az-AZ": "nort2697",
    "bn-BD": "beng1280", "ca-ES": "stan1289", "cy-GB": "wels1247", "da-DK": "dani1285",
    "de-DE": "stan1295", "el-GR": "mode1248", "en-US": "stan1293", "fa-IR": "west2369",
    "fi-FI": "finn1318", "fr-FR": "stan1290", "he-IL": "hebr1245", "hi-IN": "hind1269",
    "hu-HU": "hung1274", "hy-AM": "nucl1235", "id-ID": "indo1316", "is-IS": "icel1247",
    "it-IT": "ital1282", "jv-ID": "java1254", "ja-JP": "nucl1643", "kn-IN": "nucl1305",
    "ka-GE": "nucl1302", "km-KH": "cent1989", "ko-KR": "kore1280", "lv-LV": "latv1249",
    "ml-IN": "mala1464", "mn-MN": "mong1331", "ms-MY": "stan1306", "my-MM": "nucl1310",
    "nl-NL": "dutc1256", "nb-NO": "norw1259", "pl-PL": "poli1260", "pt-PT": "port1283",
    "ro-RO": "roma1327", "ru-RU": "russ1263", "sl-SL": "slov1268", "es-ES": "stan1288",
    "sq-AL": "gheg1238", "sw-KE": "swah1253", "sv-SE": "swed1254", "ta-IN": "tami1289",
    "te-IN": "telu1262", "tl-PH": "taga1270", "th-TH": "thai1261", "tr-TR": "nucl1301",
    "ur-PK": "urdu1245", "vi-VN": "viet1252", "zh-TW": "mand1415", "zh-CN": "mand1415"
}

# Dictionary mapping the locale codes (language-country) from SEMREL languages to their corresponding glottocodes
SEMREL_LANGS = {
    "afr": "afri1274", "amh": "amha1245", "arb": "stan1318", "arq": "alge1239",
    "ary": "moro1292", "eng": "stan1293", "spa": "stan1288", "hau": "haus1257",
    "hin": "hind1269", "ind": "indo1316", "kin": "kiny1244", "mar": "mara1378",
    "pan": "panj1256", "tel": "telu1262"
}

# Initialize the URIEL+ system and enable caching for efficiency
u = uriel.URIELPlus()
u.reset()
u.set_cache(True)
u.integrate_databases()

# Geo vectors
update_dataset_with_vectors(dataset_filename="geo.pt", vector_type="geographic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                            languages=MASAKHANEWS_LANGS)
update_dataset_with_vectors(dataset_filename="geo.pt", vector_type="geographic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                            languages=MASSIVE_LANGS)
update_dataset_with_vectors(dataset_filename="geo.pt", vector_type="geographic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                            languages=SEMREL_LANGS)

# Syntax average vectors
u.set_aggregation('A')
u.softimpute_imputation()

update_dataset_with_vectors(dataset_filename="syntax_average.pt", vector_type="syntactic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                            languages=MASAKHANEWS_LANGS)
update_dataset_with_vectors(dataset_filename="syntax_average.pt", vector_type="syntactic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                            languages=MASSIVE_LANGS)
update_dataset_with_vectors(dataset_filename="syntax_average.pt", vector_type="syntactic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                            languages=SEMREL_LANGS)

# Syntax average geo vectors
create_concatenated_vector(first_dataset_filename="syntax_average.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                           languages=MASAKHANEWS_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_average.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                           languages=MASSIVE_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_average.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                           languages=SEMREL_LANGS)

# Syntax KNN vectors
u.reset()
u.set_cache(True)
u.integrate_databases()
u.set_aggregation('U')
u.softimpute_imputation()

update_dataset_with_vectors(dataset_filename="syntax_knn.pt", vector_type="syntactic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                            languages=MASAKHANEWS_LANGS)
update_dataset_with_vectors(dataset_filename="syntax_knn.pt", vector_type="syntactic", 
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                            languages=MASSIVE_LANGS)
update_dataset_with_vectors(dataset_filename="syntax_knn.pt", vector_type="syntactic",
                            experiment_dir=os.path.join(EXPERIMENT_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                            languages=SEMREL_LANGS)

# Syntax KNN geo vectors
create_concatenated_vector(first_dataset_filename="syntax_knn.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                           languages=MASAKHANEWS_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_knn.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                           languages=MASSIVE_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_knn.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                           languages=SEMREL_LANGS)

# Syntax KNN syntax average vectors
create_concatenated_vector(first_dataset_filename="syntax_knn.pt", second_dataset_filename="syntax_average.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                           languages=MASAKHANEWS_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_knn.pt", second_dataset_filename="syntax_average.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                           languages=MASSIVE_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_knn.pt", second_dataset_filename="syntax_average.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                           languages=SEMREL_LANGS)

# Syntax KNN syntax average geo vectors
create_concatenated_vector(first_dataset_filename="syntax_knn_syntax_average.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "masakhanews_vectors"), output_dir=os.path.join(BASE_DIR, "masakhanews_vectors"),
                           languages=MASAKHANEWS_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_knn_syntax_average.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "massive_vectors"), output_dir=os.path.join(BASE_DIR, "massive_vectors"),
                           languages=MASSIVE_LANGS)

create_concatenated_vector(first_dataset_filename="syntax_knn_syntax_average.pt", second_dataset_filename="geo.pt", 
                           experiment_dir=os.path.join(BASE_DIR, "semrel_vectors"), output_dir=os.path.join(BASE_DIR, "semrel_vectors"),
                           languages=SEMREL_LANGS)