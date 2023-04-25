# transferLearningNER
Creating a robust NER tool for low resource language using training from high-resource decendant/sister languages

Sanskrit shares a lot of grammatical, phonetic and structural properties with its descendant/sibling languages.
Create an NER model trained on languages like Hindi, Telugu and Malayalam
Use transliteration as a tool to achieve better NER accuracy in Sanskrit over the models trained on Hindi, Telugu and Malayalam.

Why Transliteration?
Transliteration is a simple script change operation with small scope of error. 
Allows for easy transfer learning in multi-language setting with much less resource requirement than Translation.


Hindi Dataset Stats - 

|split	 |   Tokens	 | UniqueTokens	| UnknownTokens  |  Tag=0	|    Tag=1	|   Tag=2	|  Tag=3	|   Tag=4	| Tag=5	| Tag=6 |
|--------|-----------|--------------|----------------|----------|-----------|-----------|-----------|-----------|-------|-------|
|train	 | 22029408	 |       378006 |        	0	|18099478 |	767003	|  712403	| 686388	|  825713	| 731183	| 207240 |
|test	 |     8405	 |         2889	|      	   63	|    6996	|   263	|     239	|    257 	|   253	|   302	   | 95 |
|val	     |	304282	 |        28369	|     	 2897	|  249634	| 10549	|    9870	|   9735	|   11506 |	 10209	|  2779 |
