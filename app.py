from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np

app = Flask(__name__)

dependencies = {
    'auc_roc': AUC
}

verbose_name = {
0: {'scientific_name': 'Abelmoschus sagittifolius', 'localName': 'Drumstick', 'features': 'Anti-inflammatory, antioxidant'},
1: {'scientific_name': 'Abrus precatorius', 'localName': 'Rosary pea', 'features': 'Anti-cancer, anti-diabetic'},
2: {'scientific_name': 'Abutilon indicum', 'localName': 'Indian mallow', 'features': 'Anti-diarrheal, laxative'},
3: {'scientific_name': 'Acanthus integrifolius', 'localName': 'Spiny acanthus', 'features': 'Anti-inflammatory, wound healing'},
4: {'scientific_name': 'Acorus tatarinowii', 'localName': 'Sweet flag', 'features': 'Antispasmodic, diuretic'},
5: {'scientific_name': 'Agave americana', 'localName': 'American aloe', 'features': 'Anti-inflammatory, laxative'},
6: {'scientific_name': 'Ageratum conyzoides', 'localName': 'Whiteweed', 'features': 'Anti-bacterial, anti-fungal'},
7: {'scientific_name': 'Allium ramosum', 'localName': 'Chinese chive', 'features': 'Anti-hypertensive, anti-cholesterol'},
8: {'scientific_name': 'Alocasia macrorrhizos', 'localName': 'Taro', 'features': 'Anti-inflammatory, laxative'}, 
9: {'scientific_name': 'Aloe vera', 'localName': 'Aloe', 'features': 'Anti-inflammatory, wound healing'},
10 :{'scientific_name':'Alpinia officinarum','localName': 'Galangal', 'features': 'Anti-spasmodic, carminative'},
11 :{'scientific_name':'Amomum longiligulare','localName': 'Long-lipped cardamom', 'features': 'Anti-inflammatory, expectorant'},
12 :{'scientific_name':'Ampelopsis cantoniensis','localName': 'Five-angled grape', 'features': 'Anti-inflammatory, antioxidant'},
13 :{'scientific_name':'Andrographis paniculata','localName': 'Andrographis', 'features': 'Anti-bacterial, anti-viral'},
14 :{'scientific_name':'Angelica dahurica','localName': 'Chinese angelica', 'features': 'Anti-inflammatory, analgesic'},
15 :{'scientific_name':'Ardisia sylvestris','localName': 'Ceylon gooseberry', 'features': 'Anti-bacterial, anti-fungal'},
16 :{'scientific_name':'Artemisia vulgaris','localName': 'Mugwort', 'features': 'Anti-inflammatory, antioxidant'},
17 :{'scientific_name':'Artocarpus altilis','localName': 'Breadfruit', 'features': 'Anti-diabetic, anti-cancer'},
18 :{'scientific_name':'Artocarpus heterophyllus','localName': 'Jackfruit', 'features': 'Anti-diabetic, anti-inflammatory'},
19 :{'scientific_name':'Artocarpus lakoocha','localName': 'Breadnut', 'features': 'Anti-diabetic, anti-inflammatory'},
20 :{'scientific_name':'Asparagus cochinchinensis','localName': 'Vietnamese asparagus', 'features': 'Diuretic, laxative'},
21 :{'scientific_name':'Asparagus officinalis','localName': 'Asparagus', 'features': 'Diuretic, antioxidant'},
22 :{'scientific_name':'Averrhoa carambola','localName': 'Starfruit', 'features': 'Antioxidant, anti-inflammatory'},
23 :{'scientific_name':'Baccaurea sp.','localName': 'Wild mangosteen', 'features': 'Anti-inflammatory, antioxidant'},
24 :{'scientific_name':'Barleria lupulina','localName': 'Indian honeysuckle', 'features': 'Anti-inflammatory, wound healing'},
25 :{'scientific_name':'Bengal arum','localName': 'Arum colocasia', 'features': 'Anti-inflammatory, diuretic'},
26 :{'scientific_name':'Berchemia lineata','localName': 'Chinese privet', 'features': 'Anti-diabetic, anti-cancer'},
27 :{'scientific_name':'Bidens pilosa','localName': 'Beggar ticks', 'features': 'Anti-inflammatory, anti-cancer'},
28 :{'scientific_name':'Bischofia trifoliata','localName': 'Tree of life', 'features': 'Anti-diabetic, anti-inflammatory'},
29 :{'scientific_name':'Blackberry lily','localName': 'Himalayan blackberry', 'features': 'Anti-inflammatory, antioxidant'},
30 :{'scientific_name':'Blumea balsamifera','localName': 'Balsam flower', 'features': 'Anti-inflammatory, wound healing'},
31 :{'scientific_name':'Boehmeria nivea','localName': 'Paper mulberry', 'features': 'Anti-inflammatory, diuretic'},
32 :{'scientific_name':'Breynia vitis','localName': 'Sourleaf', 'features': 'Anti-inflammatory, laxative'},
33 :{'scientific_name':'Caesalpinia sappan','localName': 'Sappanwood', 'features': 'Anti-inflammatory, wound healing'},
34 :{'scientific_name':'Callerya speciosa','localName': 'Butterfly pea', 'features': 'Anti-diabetic, antioxidant'},
35:{'scientific_name':'Callisia fragrans','localName': 'Mexican bamboo', 'features': 'Anti-inflammatory, diuretic'},
36 :{'scientific_name':'Calophyllum inophyllum','localName': 'Tamanu oil tree', 'features': 'Anti-inflammatory, wound healing'},
37 :{'scientific_name':'Calotropis gigantea','localName': 'Crown flower', 'features': 'Anti-inflammatory, anti-cancer'},
38 :{'scientific_name':'Camellia chrysantha','localName': 'Golden camellia', 'features': 'Anti-inflammatory, antioxidant'},
39 :{'scientific_name':'Caprifoliaceae','localName': 'Honeysuckle family', 'features': 'Anti-inflammatory, sedative'},
40 :{'scientific_name':'Capsicum annuum','localName': 'Chili pepper', 'features': 'Anti-inflammatory, antioxidant'},
41 :{'scientific_name':'Carica papaya','localName': 'Papaya', 'features': 'Digestive aid, wound healing'},
42 :{'scientific_name':'Catharanthus roseus','localName': 'Madagascar periwinkle', 'features': 'Anti-cancer, anti-fungal'},
43 :{'scientific_name':'Celastrus hindsii','localName': 'Climbing bittersweet', 'features': 'Anti-inflammatory, antioxidant'},
44 :{'scientific_name':'Celosia argentea','localName': 'Cockscomb', 'features': 'Anti-inflammatory, antioxidant'},
45 :{'scientific_name':'Centella asiatica','localName': 'Gotu kola', 'features': 'Anti-inflammatory, wound healing'},
46 :{'scientific_name':'Citrus aurantifolia','localName': 'Lime', 'features': 'Digestive aid, antioxidant'},
47 :{'scientific_name':'Citrus hystrix','localName': 'Kaffir lime', 'features': 'Digestive aid, antioxidant'},
48 :{'scientific_name':'Clausena indica','localName': 'Wood apple', 'features': 'Anti-inflammatory, antioxidant'},
49  :{'scientific_name':'Cleistocalyx operculatus','localName': 'Pink shell ginger', 'features': 'Anti-inflammatory, antioxidant'},
50 :{'scientific_name':'Clerodendrum inerme','localName': 'Travellers joy', 'features': 'Anti-inflammatory, wound healing'},
51 :{'scientific_name':'Clinacanthus nutans','localName': 'Spider plant', 'features': 'Anti-inflammatory, antioxidant'},
52 :{'scientific_name':'Clycyrrhiza uralensis Fisch.','localName': 'Licorice', 'features': 'Anti-inflammatory, anti-ulcer'},
53 :{'scientific_name':'Coix lacryma-jobi','localName': 'Jobs tars', 'features': 'Anti-diabetic, diuretic'},
54 :{'scientific_name':'Cordyline fruticosa','localName': 'Ti plant', 'features': 'Anti-inflammatory, antioxidant'},
55 :{'scientific_name':'Costus speciosus','localName': 'Kansui', 'features': 'Anti-inflammatory, diuretic'},
56 :{'scientific_name':'Crescentia cujete L.','localName': 'Calabash tree', 'features': 'Anti-inflammatory, laxative'},
57 :{'scientific_name':'Crinum asiaticum','localName': 'Crinum lily', 'features': 'Anti-inflammatory, wound healing'},
58 :{'scientific_name':'Crinum latifolium','localName': 'Madagascar lily', 'features': 'Anti-inflammatory, wound healing'},
59 :{'scientific_name':'Croton oblongifolius','localName': 'Croton', 'features': 'Anti-inflammatory, laxative'},
60 :{'scientific_name':'Croton tonkinensis','localName': 'Tonkin croton', 'features': 'Anti-inflammatory, laxative'},
61 :{'scientific_name':'Curculigo gracilis','localName': 'White turmeric', 'features': 'Anti-inflammatory, antioxidant'},
62 :{'scientific_name':'Curculigo orchioides','localName': 'Black turmeric', 'features': 'Anti-inflammatory, antioxidant'},
63 :{'scientific_name':'Cymbopogon','localName': 'Lemongrass', 'features': 'Anti-inflammatory, antioxidant'},
64 :{'scientific_name':'Datura metel','localName': 'Thorn apple', 'features': 'Antispasmodic, sedative'},
65 :{'scientific_name':'Derris elliptica','localName': 'Derris', 'features': 'Antibacterial, anti-fungal'},
66 :{'scientific_name':'Dianella ensifolia','localName': 'Blue flax lily', 'features': 'Anti-inflammatory, diuretic'},
67 :{'scientific_name':'Dicliptera chinensis','localName': 'Chinese water pepper', 'features': 'Anti-inflammatory, diuretic'},
68 :{'scientific_name':'Dimocarpus longan','localName': 'Longan', 'features': 'Anti-aging, antioxidant'},
69 :{'scientific_name':'Dioscorea persimilis','localName': 'Wild yam', 'features': 'Anti-inflammatory, antioxidant'},
70 :{'scientific_name':'Eichhornia crassipes','localName': 'Water hyacinth', 'features': 'Anti-inflammatory, wound healing'},
71 :{'scientific_name':'Eleutherine bulbosa','localName': 'Turmeric bulb', 'features': 'Anti-inflammatory, antioxidant'},
74 :{'scientific_name':'Eupatorium triplinerve','localName': 'Thoroughwort', 'features': 'Anti-inflammatory, antioxidant'},
75 :{'scientific_name':'Euphorbia hirta','localName': 'Croton tiglium', 'features': 'Anti-inflammatory, laxative'},
72 :{'scientific_name':'Erythrina variegata','localName': 'Coral tree', 'features': 'Anti-inflammatory, antioxidant'},
76 :{'scientific_name':'Euphorbia pulcherrima','localName': 'Poinsettia', 'features': 'Anti-inflammatory, antioxidant'},
73 :{'scientific_name':'Eupatorium fortunei','localName': 'Boneset', 'features': 'Anti-inflammatory, antioxidant'},
77 :{'scientific_name':'Euphorbia tirucalli','localName': 'Pencil cactus', 'features': 'Anti-inflammatory, laxative'},
78 :{'scientific_name':'Euphorbia tithymaloides','localName': 'Firecracker plant', 'features': 'Anti-inflammatory, laxative'},
79 :{'scientific_name':'Eurycoma longifolia','localName': 'Tongkat ali', 'features': 'Anti-aging, antioxidant'},
80 :{'scientific_name':'Excoecaria cochinchinensis','localName': 'Cochin china hog plum', 'features': 'Anti-inflammatory, antioxidant'},
81 :{'scientific_name':'Excoecaria sp','localName': 'Excoecaria', 'features': 'Anti-inflammatory, antioxidant'},
82 :{'scientific_name':'Fallopia multiflora','localName': 'Japanese knotweed', 'features': 'Anti-inflammatory, antioxidant'},
84 :{'scientific_name':'Ficus racemosa','localName': 'Banyan tree', 'features': 'Anti-inflammatory, antioxidant'},
83 :{'scientific_name':'Ficus auriculata','localName': 'Golden fig', 'features': 'Anti-inflammatory, antioxidant'},
85 :{'scientific_name':'Fructus lycii','localName': 'Lycium berry', 'features': 'Antioxidant, anti-aging'},
86 :{'scientific_name':'Glochidion eriocarpum','localName': 'Hairy leaved spurge', 'features': 'Anti-inflammatory, laxative'},
87 :{'scientific_name':'Glycosmis pentaphylla','localName': 'Cape gooseberry', 'features': 'Anti-inflammatory, antioxidant'},
89 :{'scientific_name':'Gymnema sylvestre','localName': 'Gurmar', 'features': 'Anti-diabetic, antioxidant'},
88 :{'scientific_name':'Gonocaryum lobbianum','localName': 'Indian mulberry', 'features': 'Anti-inflammatory, antioxidant'},
91 :{'scientific_name':'Hemerocallis fulva','localName': 'Daylily', 'features': 'Anti-inflammatory, antioxidant'},
90 :{'scientific_name':'Gynura divaricata','localName': 'Heartleaf bitter gourd', 'features': 'Anti-cancer, antioxidant'},
92 :{'scientific_name':'Hemigraphis glaucescens','localName': 'Spiderleaf', 'features': 'Anti-inflammatory, wound healing'},
93 :{'scientific_name':'Hibiscus mutabilis','localName': 'Changeable roselle', 'features': 'Anti-inflammatory, antioxidant'},
94 :{'scientific_name':'Hibiscus rosa-sinensis','localName': 'Chinese hibiscus', 'features': 'Anti-inflammatory, antioxidant'},
95 :{'scientific_name':'Hibiscus sabdariffa','localName': 'Roselle', 'features': 'Anti-inflammatory, antioxidant'},
96 :{'scientific_name':'Holarrhena pubescens','localName': 'Devils pepper', 'features': 'Anti-inflammatory, anti-malarial'},
# 97 :{'scientific_name':'Holarrhena antidysenterica','localName': 'Kurchi', 'features': 'Anti-inflammatory, anti-malarial'},
97 :{'scientific_name':'Homalomena occulta','localName': 'Elephant ear', 'features': 'Anti-inflammatory, diuretic'},
98 :{'scientific_name':'Houttuynia cordata','localName': 'Heartleaf houttuynia', 'features': 'Anti-inflammatory, antioxidant'},
99 :{'scientific_name':'Imperata cylindrica','localName': 'Imperata', 'features': 'Anti-inflammatory, diuretic'},
100:{'scientific_name':'Iris domestica','localName': 'Japanese iris', 'features': 'Anti-inflammatory, antioxidant'},
101:{'scientific_name':'Ixora coccinea','localName': 'Ixora', 'features': 'Anti-inflammatory, antioxidant'},
102:{'scientific_name':'Jasminum sambac','localName': 'Arabian jasmine', 'features': 'Anti-inflammatory, sedative'},
103:{'scientific_name':'Jatropha gossypiifolia','localName': 'Physic nut', 'features': 'Anti-inflammatory, laxative'},
104:{'scientific_name':'Jatropha multifida','localName': 'Coral plant', 'features': 'Anti-inflammatory, laxative'},
105:{'scientific_name':'Jatropha podagrica','localName': 'Elephants foot', 'features': 'Anti-inflammatory, laxative'},
106:{'scientific_name':'Justicia gendarussa','localName': 'Gandarusa', 'features': 'Anti-inflammatory, laxative'},
107:{'scientific_name':'Kalanchoe pinnata','localName': 'Air plant', 'features': 'Anti-inflammatory, wound healing'},
108:{'scientific_name':'Lactuca indica','localName': 'Indian lettuce', 'features': 'Anti-inflammatory, antioxidant'},
109:{'scientific_name':'Lantana camara','localName': 'Lantana', 'features': 'Anti-bacterial, antioxidant'},
110:{'scientific_name':'Lawsonia inermis','localName': 'Henna', 'features': 'Anti-inflammatory, antioxidant'},
111:{'scientific_name':'Leea rubra','localName': 'Red gooseberry', 'features': 'Anti-inflammatory, antioxidant'},
112:{'scientific_name':'Litsea glutinosa','localName': 'Litsea', 'features': 'Anti-inflammatory, antioxidant'},
113:{'scientific_name':'Lonicera dasystyla','localName': 'Honeysuckle', 'features': 'Anti-inflammatory, antioxidant'},
114:{'scientific_name':'Lpomoea sp','localName': 'Morning glory', 'features': 'Anti-inflammatory, antioxidant'},
115:{'scientific_name':'Maesa','localName': 'Maesa', 'features': 'Anti-inflammatory, antioxidant'},
116:{'scientific_name':'Mallotus barbatus','localName': 'Mallotus', 'features': 'Anti-inflammatory, antioxidant'},
117:{'scientific_name':'Mangifera','localName': 'Mango', 'features': 'Anti-inflammatory, antioxidant'},
118:{'scientific_name':'Melastoma malabathricum','localName': 'Indian rhododendron', 'features': 'Anti-inflammatory, antioxidant'},
119:{'scientific_name':'Mentha spicata','localName': 'Spearmint', 'features': 'Anti-inflammatory, antioxidant'},
120:{'scientific_name':'Microcos tomentosa','localName': 'Sea grape', 'features': 'Anti-inflammatory, antioxidant'},
121:{'scientific_name':'Micromelum falcatum','localName': 'Bay rum tree', 'features': 'Astringent,antiseptic'},
122:{'scientific_name': 'Millettia pulchra', 'localName': 'Pink shower', 'features': 'Anti-inflammatory, antioxidant'},
123: {'scientific_name': 'Mimosa pudica', 'localName': 'Sensitive plant', 'features': 'Anti-inflammatory, wound healing'},
124: {'scientific_name': 'Morinda citrifolia', 'localName': 'Noni', 'features': 'Anti-inflammatory, antioxidant'},
125: {'scientific_name': 'Moringa oleifera', 'localName': 'Drumstick tree', 'features': 'Anti-inflammatory, antioxidant'},
126:{'scientific_name': 'Morus alba', 'localName': 'Mulberry', 'features': 'Anti-inflammatory, antioxidant'},
127:{'scientific_name': 'Mussaenda philippica', 'localName': 'Pink Mussaenda', 'features': 'Anti-inflammatory, antioxidant'},
128:{'scientific_name': 'Nelumbo nucifera', 'localName': 'Sacred lotus','features': 'Anti-inflammatory, antioxidant'},
129:{'scientific_name': 'Ocimum basilicum', 'localName': 'Basil','features': 'Anti-inflammatory, antioxidant'},
130:{'scientific_name': 'Ocimum gratissimum', 'localName': 'Holy basil','features': 'Anti-inflammatory, antioxidant'},
131:{'scientific_name': 'Ocimum sanctum', 'localName': 'Tulsi','features': 'Anti-inflammatory, antioxidant'},
132: {'scientific_name': 'Oenanthe javanica', 'localName': 'Javanese water dropwort','features': 'Anti-inflammatory, diuretic'},
133: {'scientific_name': 'Ophiopogon japonicus', 'localName': 'Japanese lilyturf','features': 'Anti-inflammatory, antioxidant'},
134:{'scientific_name': 'Paederia lanuginosa', 'localName': 'Cats whiskers','features': 'Anti-inflammatory, antioxidant'},
135:{'scientific_name': 'Pandanus amaryllifolius', 'localName': 'Screwpine','features': 'Anti-inflammatory, antioxidant'},
136:{'scientific_name': 'Pandanus sp.', 'localName': 'Screwpine','features': 'Anti-inflammatory, antioxidant'},
137:{'scientific_name': 'Pandanus tectorius', 'localName': 'Screwpine','features': 'Anti-inflammatory, antioxidant'},
138: {'scientific_name': 'Parameria laevigata', 'localName': 'Parameria','features': 'Anti-inflammatory, antioxidant'},
139: {'scientific_name': 'Passiflora foetida', 'localName': 'Stinking passionflower','features': 'Anti-inflammatory, sedative'},
140: {'scientific_name': 'Pereskia sacharosa', 'localName': 'Sweet pitahaya','features': 'Anti-diabetic, antioxidant'},
141:{'scientific_name': 'Persicaria odorata', 'localName': 'Water pepper','features': 'Anti-inflammatory, diuretic'},
142:{'scientific_name': 'Phlogacanthus turgidus', 'localName': 'Firespike','features': 'Anti-inflammatory, antioxidant'},
143:{'scientific_name': 'Phrynium placentarium', 'localName': 'Elephants ear','features': 'Anti-inflammatory, diuretic'},
144:{'scientific_name': 'Phyllanthus reticulatus', 'localName': 'Toothache plant','features': 'Anti-inflammatory, antibacterial'},
145:{'scientific_name': 'Piper betle', 'localName': 'Betel leaf','features': 'Antimicrobial, antioxidant'},
146:{'scientific_name': 'Piper sarmentosum', 'localName': 'Cats whiskers','features': 'Anti-inflammatory, antioxidant'},
147:{'scientific_name': 'Plantago', 'localName': 'Plantain','features': 'Anti-inflammatory, wound healing'},
148:{'scientific_name': 'Platycladus orientalis', 'localName': 'Chinese arborvitae','features': 'Anti-inflammatory, antioxidant'},
149:{'scientific_name': 'Plectranthus amboinicus', 'localName': 'Cuban oregano','features': 'Anti-inflammatory, antioxidant'},
150:{'scientific_name': 'Pluchea pteropoda Hemsl', 'localName': 'Pluchea','features': 'Anti-inflammatory, antioxidant'},
151:{'scientific_name': 'Plukenetia volubilis', 'localName': 'Sacha inchi','features': 'Anti-inflammatory, antioxidant'},
152:{'scientific_name': 'Plumbago indica', 'localName': 'Leadwort','features': 'Anti-inflammatory, antioxidant'},
153:{'scientific_name': 'Plumeria rubra', 'localName': 'Frangipani','features': 'Anti-inflammatory, sedative'},
154:{'scientific_name': 'Polyginum cuspidatum', 'localName': 'Japanese knotweed','features': 'Anti-inflammatory, antioxidant'},
155:{'scientific_name': 'Polyscias fruticosa', 'localName': 'Ming aralia','features': 'Anti-inflammatory, antioxidant'},
156:{'scientific_name': 'Polyscias guilfoylei', 'localName': 'Ming aralia','features': 'Anti-inflammatory, antioxidant'},
157:{'scientific_name': 'Polyscias scutellaria', 'localName': 'Ming aralia','features': 'Anti-inflammatory, antioxidant'},
158:{'scientific_name': 'Pouzolzia zeylanica', 'localName': 'Climbing fig','features': 'Anti-inflammatory, diuretic'},
159:{'scientific_name': 'Premna serratifolia', 'localName': 'Ceylon leadwort','features': 'Anti-inflammatory, antioxidant'},
160:{'scientific_name': 'Pseuderanthemum latifolium', 'localName': 'Indian nightshade','features':'Anti-inflammatory, antioxidant'},
161:{'scientific_name': 'Psidium guajava', 'localName': 'Guava','features': 'Anti-inflammatory, antioxidant'},
162:{'scientific_name': 'Psychotria reevesii Wall', 'localName': 'Cats claw','features': 'Anti-inflammatory, antioxidant'},
163:{'scientific_name': 'Psychotria rubra', 'localName': 'Brazilian pepper','features': 'Anti-inflammatory, antioxidant'},
164:{'scientific_name': 'Quisqualis indica', 'localName': 'Rangoon creeper','features': 'Anti-inflammatory, antioxidant'},
165:{'scientific_name': 'Rauvolfia', 'localName': 'Serpentine','features': 'Anti-hypertensive, sedative'},
166:{'scientific_name': 'Rauvolfia tetraphylla', 'localName': 'Indian snakeroot','features': 'Anti-hypertensive, sedative'},
167:{'scientific_name': 'Rhinacanthus nasutus', 'localName': 'Indian mallow','features': 'Anti-inflammatory, wound healing'},
168:{'scientific_name': 'Rhodomyrtus tomentosa', 'localName': 'Rose apple','features': 'Anti-inflammatory, antioxidant'},
169:{'scientific_name': 'Ruellia tuberosa', 'localName': 'Texas star hibiscus','features': 'Anti-inflammatory, antioxidant'},
170:{'scientific_name': 'Sanseviera canaliculata Carr', 'localName': 'Snake plant','features': 'Anti-inflammatory, wound healing'},
171:{'scientific_name': 'Sansevieria hyacinthoides', 'localName':'Snake Plant', 'features': 'Anti-inflammatory, antioxidant'},
172:{'scientific_name': 'Sarcandra glabra', 'localName': 'Chinese sarsaparilla','features': 'Anti-inflammatory, antioxidant'},
173:{'scientific_name': 'Sauropus androgynus', 'localName': 'Prickly sida','features': 'Anti-inflammatory, antioxidant'},
174:{'scientific_name': 'Schefflera heptaphylla', 'localName': 'Umbrella tree','features': 'Anti-inflammatory, antioxidant'},
175:{'scientific_name': 'Schefflera venulosa', 'localName': 'Ming aralia','features': 'Anti-inflammatory, antioxidant'},
176:{'scientific_name': 'Senna alata', 'localName': 'Egyptian cassia','features': 'Anti-inflammatory, laxative'},
178:{'scientific_name': 'Solanum mammosum', 'localName': 'Prickly pear','features': 'Anti-inflammatory, antioxidant'},
179:{'scientific_name': 'Solanum torvum', 'localName': 'Horse nettle','features': 'Anti-inflammatory, antioxidant'},
177:{'scientific_name': 'Sida acuta Burm', 'localName': 'Prickly sida','features': 'Anti-inflammatory, antioxidant'},
180:{'scientific_name': 'Spilanthes acmella', 'localName': 'Toothache plant','features': 'Anti-inflammatory, antibacterial'},
181:{'scientific_name': 'Spondias dulcis', 'localName': 'Hog plum','features': 'Anti-inflammatory, antioxidant'},
182:{'scientific_name': 'Stachytarpheta jamaicensis', 'localName': 'Spanish needles','features': 'Anti-inflammatory, antioxidant'},
183:{'scientific_name': 'Stephania dielsiana', 'localName': 'Climbing moonseed','features': 'Anti-inflammatory, antioxidant'},
184:{'scientific_name': 'Stereospermum chelonoides', 'localName': 'Trumpet flower','features': 'Anti-inflammatory, antioxidant'},
185:{'scientific_name': 'Streptocaulon juventas', 'localName': 'Creeping buttercup','features': 'Anti-inflammatory, antioxidant'},
186:{'scientific_name': 'Syzygium nervosum', 'localName': 'Cluster fig','features': 'Anti-inflammatory, antioxidant'},
187:{'scientific_name': 'Tabernaemontana divaricata', 'localName': 'Yellow oleander','features': 'Anti-inflammatory, antioxidant'},
188:{'scientific_name': 'Tacca subflabellata', 'localName': 'Black bat flower','features': 'Anti-inflammatory, antioxidant'},
189:{'scientific_name': 'Tamarindus indica', 'localName': 'Tamarind','features': 'Anti-inflammatory, antioxidant'},
190:{'scientific_name': 'Terminalia catappa', 'localName': 'Indian almond','features': 'Anti-inflammatory, antioxidant'},
191:{'scientific_name': 'Tradescantia discolor', 'localName': 'Wandering Jew','features': 'Anti-inflammatory, antioxidant'},
192:{'scientific_name': 'Trichanthera gigantea', 'localName': 'Giant tickweed','features': 'Anti-inflammatory, antioxidant'},
193:{'scientific_name': 'Vernonia amygdalina', 'localName': 'Crape myrtle','features': 'Anti-inflammatory, antioxidant'},
194:{'scientific_name': 'Vitex negundo', 'localName': 'Chaste tree','features': 'Anti-inflammatory, antioxidant'},
195:{'scientific_name': 'Xanthium strumarium', 'localName': 'Cocklebur','features': 'Anti-inflammatory, antioxidant'},
196:{'scientific_name': 'Zanthoxylum avicennae', 'localName': 'Toothache tree','features': 'Anti-inflammatory, antioxidant'},
197:{'scientific_name': 'Zingiber officinale', 'localName': 'Ginger','features': 'Anti-inflammatory, antioxidant'},
198:{'scientific_name': 'Ziziphus mauritiana', 'localName': 'Jujube','features': 'Anti-inflammatory, antioxidant'},
199:{'scientific_name': 'Helicteres hirsuta', 'localName': 'Golden shower','features': 'Anti-inflammatory, antioxidant'}
}
# Select model


model = load_model('plant.h5')

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(180, 180))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = test_image.reshape(1, 180, 180, 3)

    predict_x = model.predict(test_image)
    classes_x = np.argmax(predict_x, axis=1)
    
    predicted_plant = list(verbose_name.keys())[classes_x[0]]
    return verbose_name[predicted_plant]

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/tests/" + img.filename
  	
		img.save(img_path)

		predict_result = predict_label(img_path)

	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
	return render_template('performance.html')
    
@app.route("/chart")
def chart():
	return render_template('chart.html') 

	
if __name__ =='__main__':
	app.run(debug = True)