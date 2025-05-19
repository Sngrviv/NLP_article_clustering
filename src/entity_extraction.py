"""
Entity extraction module for Marathi news articles.
This module extracts entities like places, emotions, and severity from text.
"""

import re
import nltk
from collections import Counter

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define emotion keywords in Marathi with weighted importance
EMOTION_KEYWORDS = {
    'positive': [
        ('आनंद', 2), ('खुश', 2), ('समाधान', 1.5), ('उत्साह', 1.5), ('प्रेम', 1.5), 
        ('आशा', 1), ('विश्वास', 1), ('सुख', 2), ('हर्ष', 1.5), ('उल्हास', 1), 
        ('उत्सव', 1), ('स्वागत', 0.8), ('यश', 1.5), ('जय', 1), ('विजय', 1), ('चांगले', 0.7)
    ],
    'negative': [
        ('दुःख', 2), ('राग', 2), ('भय', 1.8), ('चिंता', 1.5), ('निराशा', 1.5), 
        ('दु:ख', 2), ('त्रास', 1.3), ('संताप', 1.5), ('क्रोध', 2), ('भीती', 1.8), 
        ('धोका', 1), ('अपयश', 1.2), ('हानी', 1), ('नुकसान', 1), ('समस्या', 0.8), 
        ('वाईट', 0.7), ('धक्कादायक', 1.5)
    ],
    'neutral': [
        ('माहिती', 1), ('सूचना', 0.8), ('जाहीर', 0.8), ('घोषणा', 0.8), ('निर्णय', 0.8), 
        ('नियोजन', 0.8), ('व्यवस्था', 0.8), ('प्रक्रिया', 0.8), ('कार्यक्रम', 0.5), 
        ('योजना', 0.8), ('अभ्यास', 0.8), ('निरीक्षण', 1), ('तपासणी', 0.8)
    ]
}

# Define severity keywords in Marathi with weighted importance
SEVERITY_KEYWORDS = {
    'high': [
        ('गंभीर', 2), ('तीव्र', 1.8), ('मोठा', 1), ('भयंकर', 2), ('प्रचंड', 1.8), 
        ('अत्यंत', 1.5), ('अतिशय', 1.5), ('जबरदस्त', 1.5), ('हाहाकार', 2), 
        ('संकट', 1.8), ('आपत्ती', 2), ('दुर्घटना', 2), ('मृत्यू', 2.5), 
        ('मरण', 2.5), ('जीवितहानी', 2.5), ('धोका', 1.5)
    ],
    'medium': [
        ('मध्यम', 1.5), ('सामान्य', 1), ('काही प्रमाणात', 1), ('थोडा', 0.8), 
        ('थोडेसे', 0.8), ('प्रमाणात', 0.8), ('सौम्य', 1), ('किंचित', 0.8), 
        ('जखमी', 1.5), ('त्रास', 1.2), ('अडचण', 1.2), ('समस्या', 1.2)
    ],
    'low': [
        ('किरकोळ', 1.5), ('सौम्य', 1), ('हलका', 1.2), ('कमी', 0.8), ('अल्प', 0.8), 
        ('नगण्य', 1), ('छोटा', 0.7), ('थोडा', 0.7), ('सुधारणा', 1), ('आराम', 0.8), 
        ('नियंत्रण', 1), ('सुरक्षित', 1.5), ('सुरळीत', 1.2)
    ]
}

# Define category keywords in Marathi with weighted importance
CATEGORY_KEYWORDS = {
    'Technology': [
        ('तंत्रज्ञान', 3), ('मोबाईल', 2), ('कंप्युटर', 2), ('इंटरनेट', 2), ('सॉफ्टवेअर', 2), 
        ('हार्डवेअर', 2), ('अॅप', 1.8), ('वेबसाईट', 1.5), ('डिजिटल', 1.8), ('ऑनलाइन', 1.5), 
        ('स्मार्टफोन', 2), ('गॅजेट', 1.8), ('आयटी', 2.5), ('टेक', 2.5), ('रोबोट', 1.5),
        ('आर्टिफिशियल इंटेलिजन्स', 2.5), ('एआय', 2.5), ('क्लाउड', 1.8), ('सायबर', 1.8), 
        ('इलेक्ट्रॉनिक', 1.5), ('इनोव्हेशन', 1.5), ('वायफाय', 1.5), ('नेटवर्क', 1.5), 
        ('प्रोग्रामिंग', 2), ('कोडिंग', 2), ('डेटा', 1.5), ('सिस्टम', 1), ('अपडेट', 1), 
        ('अपग्रेड', 1), ('लॅपटॉप', 1.8), ('टॅबलेट', 1.8), ('चिप', 1.5), ('प्रोसेसर', 1.5), 
        ('मेमरी', 1.5), ('स्टोरेज', 1.5), ('बॅटरी', 1.2), ('चार्जिंग', 1.2), ('ब्लूटूथ', 1.5), 
        ('5जी', 2), ('4जी', 1.8), ('वायरलेस', 1.5), ('सेन्सर', 1.5), ('कॅमेरा', 1.2), 
        ('रिसर्च', 1), ('डेव्हलपमेंट', 1)
    ],
    'Entertainment': [
        ('चित्रपट', 3), ('सिनेमा', 3), ('संगीत', 2.5), ('नाटक', 2), ('कलाकार', 1.8), 
        ('अभिनेता', 2), ('अभिनेत्री', 2), ('गाणे', 2), ('मनोरंजन', 3), ('टीव्ही', 2), 
        ('शो', 1.8), ('सिरियल', 1.8), ('डान्स', 1.5), ('नृत्य', 1.5), ('गायक', 1.8), 
        ('गायिका', 1.8), ('कला', 1.2), ('बॉलिवूड', 2.5), ('हॉलिवूड', 2), 
        ('मराठी चित्रपट', 2.5), ('वेब सिरीज', 2), ('ओटीटी', 2), ('नेटफ्लिक्स', 2), 
        ('अॅमेझॉन प्राईम', 2), ('हॉटस्टार', 2), ('यूट्यूब', 1.8), ('व्हिडिओ', 1.5), 
        ('स्ट्रीमिंग', 1.8), ('म्यूझिक', 2), ('अल्बम', 1.5), ('कॉन्सर्ट', 1.8), ('पार्टी', 1), 
        ('सेलिब्रिटी', 1.8), ('स्टार', 1.8), ('फॅशन', 1.2), ('ट्रेंड', 1), ('पुरस्कार', 1.5), 
        ('अवॉर्ड', 1.5), ('फिल्मफेअर', 1.8), ('ऑस्कर', 1.8), ('ग्रॅमी', 1.8), 
        ('डायरेक्टर', 1.8), ('प्रोड्युसर', 1.8), ('कथा', 1), ('गोष्ट', 1), ('कॉमेडी', 1.8), 
        ('ड्रामा', 1.8), ('थ्रिलर', 1.8), ('रोमँटिक', 1.5)
    ],
    'Sports': [
        ('क्रिकेट', 3), ('फुटबॉल', 3), ('हॉकी', 2.5), ('टेनिस', 2.5), ('बॅडमिंटन', 2.5), 
        ('कबड्डी', 2.5), ('खेळ', 2.8), ('खेळाडू', 2.5), ('स्पर्धा', 2), ('सामना', 2), 
        ('विजेता', 1.5), ('पराभव', 1.5), ('संघ', 1.5), ('प्रशिक्षण', 1.5), ('कोच', 2), 
        ('मैदान', 1.5), ('स्टेडियम', 2), ('ऑलिम्पिक', 2.5), ('चॅम्पियनशिप', 2.5), 
        ('मेडल', 2), ('पदक', 2), ('आयपीएल', 2.5), ('वर्ल्ड कप', 2.5), ('टी-२०', 2.5), 
        ('वनडे', 2.5), ('टेस्ट', 2.5), ('गोल', 2), ('रन', 2), ('विकेट', 2.5), ('बॉल', 1.5), 
        ('बॅट', 1.5), ('फळी', 1.5), ('धावा', 2), ('कर्णधार', 2), ('कॅप्टन', 2), 
        ('फिटनेस', 1.5), ('व्यायाम', 1), ('जिम', 1), ('ट्रेनिंग', 1.5), ('अभ्यास', 1), 
        ('प्रॅक्टिस', 1.5), ('पुरस्कार', 1.5), ('ट्रॉफी', 2), ('चषक', 2), ('जेतेपद', 2), 
        ('रेफरी', 1.5), ('पंच', 1.5), ('निर्णायक', 1.5), ('फाइनल', 2), ('सेमीफायनल', 2)
    ],
    'Politics': [
        ('राजकारण', 3), ('निवडणूक', 2.8), ('मतदान', 2.5), ('पक्ष', 2.5), ('आमदार', 2.5), 
        ('खासदार', 2.5), ('मंत्री', 2.5), ('मुख्यमंत्री', 2.8), ('पंतप्रधान', 2.8), 
        ('सरकार', 2.8), ('विरोधी पक्ष', 2.5), ('संसद', 2.5), ('विधानसभा', 2.5), 
        ('राज्यसभा', 2.5), ('लोकसभा', 2.5), ('राजकीय', 2.8), ('नेता', 2.2), ('घोषणा', 1.5), 
        ('धोरण', 1.8), ('कायदा', 2), ('राष्ट्रपती', 2.5), ('राज्यपाल', 2.5), ('प्रशासन', 1.8), 
        ('अधिकारी', 1.5), ('न्यायालय', 2), ('न्यायाधीश', 2), ('कोर्ट', 2), 
        ('सुप्रीम कोर्ट', 2.2), ('हायकोर्ट', 2.2), ('पोलीस', 1.5), ('सुरक्षा', 1.5), 
        ('दहशतवाद', 2), ('आंदोलन', 2), ('निषेध', 1.8), ('मोर्चा', 2), ('रॅली', 2), 
        ('भाषण', 1.8), ('जाहीरनामा', 2), ('मॅनिफेस्टो', 2), ('मतदार', 2.2), ('मतपेटी', 2), 
        ('ईव्हीएम', 2), ('उमेदवार', 2.2), ('प्रचार', 2), ('जनता', 1.5), ('नागरिक', 1.5)
    ],
    'Business': [
        ('व्यवसाय', 3), ('उद्योग', 2.8), ('कंपनी', 2.5), ('बाजार', 2.5), ('शेअर', 2.5), 
        ('अर्थव्यवस्था', 2.8), ('बँक', 2.5), ('वित्त', 2.5), ('गुंतवणूक', 2.5), ('नफा', 2.2), 
        ('तोटा', 2.2), ('व्यापार', 2.5), ('आयात', 2.2), ('निर्यात', 2.2), ('करार', 1.8), 
        ('भागीदारी', 1.8), ('स्टार्टअप', 2.5), ('उद्योजक', 2.5), ('व्यवस्थापन', 2),
        ('बिझनेस', 3), ('कर्ज', 2), ('लोन', 2), ('ब्याज', 1.8), ('महागाई', 2.2), 
        ('इन्फ्लेशन', 2.2), ('जीडीपी', 2.5), ('अर्थसंकल्प', 2.8), ('बजेट', 2.5), ('कर', 1.8), 
        ('टॅक्स', 1.8), ('जीएसटी', 2.2), ('आयकर', 2.2), ('सेन्सेक्स', 2.5), ('निफ्टी', 2.5), 
        ('शेअर मार्केट', 2.8), ('स्टॉक एक्सचेंज', 2.8), ('बीएसई', 2.5), ('एनएसई', 2.5), 
        ('रोजगार', 2), ('नोकरी', 2), ('पगार', 1.8), ('वेतन', 1.8), ('बोनस', 1.5), 
        ('कामगार', 1.8), ('कर्मचारी', 1.8), ('मालक', 1.8), ('सीईओ', 2.2), ('डायरेक्टर', 2)
    ],
    'Health': [
        ('आरोग्य', 3), ('रुग्णालय', 2.8), ('डॉक्टर', 2.5), ('औषध', 2.5), ('उपचार', 2.5), 
        ('रोग', 2.5), ('आजार', 2.5), ('लस', 2.5), ('वैद्यकीय', 2.8), ('मेडिकल', 2.8), 
        ('कोरोना', 2.8), ('कोविड', 2.8), ('महामारी', 2.8), ('साथीचा रोग', 2.5), 
        ('वायरस', 2.5), ('जंतू', 2), ('बॅक्टेरिया', 2.2), ('तापमान', 1.8), ('ताप', 2), 
        ('खोकला', 2), ('सर्दी', 2), ('थंडी', 1.8), ('शस्त्रक्रिया', 2.5), ('ऑपरेशन', 2.5), 
        ('तपासणी', 2.2), ('चाचणी', 2.2), ('टेस्ट', 2), ('रक्त', 2), ('हृदय', 2.2), 
        ('फुफ्फुस', 2.2), ('मेंदू', 2.2), ('पोट', 1.8), ('आतडे', 1.8), ('यकृत', 2), 
        ('मूत्रपिंड', 2), ('किडनी', 2), ('हाड', 1.8), ('स्नायू', 1.8), ('त्वचा', 1.8), 
        ('डोळे', 1.8), ('कान', 1.8), ('नाक', 1.8), ('तोंड', 1.8), ('दात', 1.8), 
        ('जिभ', 1.5), ('घसा', 1.8), ('पाठ', 1.5), ('मान', 1.5), ('पाय', 1.5), ('हात', 1.5), 
        ('बोट', 1.2), ('नख', 1), ('केस', 1), ('रक्तदाब', 2.2)
    ],
    'Education': [
        ('शिक्षण', 3), ('शाळा', 2.8), ('कॉलेज', 2.8), ('विद्यापीठ', 2.8), ('विद्यार्थी', 2.5), 
        ('शिक्षक', 2.5), ('प्राध्यापक', 2.5), ('अभ्यासक्रम', 2.5), ('परीक्षा', 2.5), 
        ('प्रवेश', 2.2), ('अॅडमिशन', 2.2), ('पदवी', 2.5), ('डिग्री', 2.5), ('पदविका', 2.5), 
        ('डिप्लोमा', 2.5), ('प्रमाणपत्र', 2), ('सर्टिफिकेट', 2), ('शिष्यवृत्ती', 2.2), 
        ('स्कॉलरशिप', 2.2), ('पुस्तक', 2), ('नोट्स', 2), ('वही', 1.5), ('पेन', 1.2), 
        ('पेन्सिल', 1.2), ('शाळा मंडळ', 2.5), ('बोर्ड', 2.2), ('सीबीएसई', 2.5), 
        ('आयसीएसई', 2.5), ('एसएससी', 2.5), ('एचएससी', 2.5), ('बीए', 2.5), ('बीएससी', 2.5), 
        ('बीकॉम', 2.5), ('बीटेक', 2.5), ('एमए', 2.5), ('एमएससी', 2.5), ('एमकॉम', 2.5), 
        ('एमटेक', 2.5), ('पीएचडी', 2.8), ('डॉक्टरेट', 2.8), ('संशोधन', 2.2), ('प्रकल्प', 2), 
        ('प्रोजेक्ट', 2), ('सेमिनार', 2), ('वर्कशॉप', 2), ('कार्यशाळा', 2)
    ],
    'Environment': [
        ('पर्यावरण', 3), ('निसर्ग', 2.8), ('वातावरण', 2.5), ('हवामान', 2.8), ('प्रदूषण', 2.8), 
        ('वायु प्रदूषण', 2.8), ('जल प्रदूषण', 2.8), ('ध्वनी प्रदूषण', 2.8), ('कचरा', 2.2), 
        ('प्लास्टिक', 2.2), ('रिसायकलिंग', 2.5), ('पुनर्वापर', 2.5), ('वृक्ष', 2.5), 
        ('झाडे', 2.5), ('जंगल', 2.5), ('वन', 2.5), ('अरण्य', 2.5), ('वन्यजीव', 2.5), 
        ('प्राणी', 2), ('पक्षी', 2), ('सागर', 2.2), ('समुद्र', 2.2), ('नदी', 2.2), 
        ('तलाव', 2.2), ('पाणी', 2.2), ('पाऊस', 2.2), ('वर्षा', 2.2), ('ऊर्जा', 2.5), 
        ('सौर ऊर्जा', 2.8), ('पवन ऊर्जा', 2.8), ('जैव ईंधन', 2.5), ('बायोफ्युएल', 2.5), 
        ('ग्लोबल वॉर्मिंग', 2.8), ('जागतिक तापमान', 2.8), ('क्लायमेट चेंज', 2.8), 
        ('हवामान बदल', 2.8), ('हरितगृह', 2.5), ('ग्रीनहाऊस', 2.5), ('ओझोन', 2.5), 
        ('वनस्पती', 2.2), ('जैवविविधता', 2.5), ('बायोडायव्हर्सिटी', 2.5)
    ],
    'Crime': [
        ('गुन्हा', 3), ('अपराध', 3), ('चोरी', 2.8), ('दरोडा', 2.8), ('खून', 3), 
        ('हत्या', 3), ('बलात्कार', 3), ('लैंगिक अत्याचार', 3), ('विनयभंग', 2.8), 
        ('छेडछाड', 2.8), ('अपहरण', 2.8), ('पळवून नेणे', 2.8), ('मारहाण', 2.5), 
        ('हल्ला', 2.5), ('धमकी', 2.5), ('फसवणूक', 2.8), ('घोटाळा', 2.8), ('भ्रष्टाचार', 2.8), 
        ('गैरव्यवहार', 2.8), ('अवैध', 2.5), ('बेकायदेशीर', 2.5), ('तस्करी', 2.8), 
        ('पोलीस', 2), ('न्यायालय', 2), ('तुरुंग', 2.5), ('कारागृह', 2.5), ('कैदी', 2.5), 
        ('आरोपी', 2.5), ('साक्षीदार', 2.2), ('पीडित', 2.5), ('न्यायाधीश', 2), 
        ('वकील', 2), ('सुनावणी', 2.2), ('अटक', 2.5), ('जामीन', 2.2)
    ],
    'Agriculture': [
        ('शेती', 3), ('शेतकरी', 3), ('पीक', 2.8), ('कृषी', 3), ('फळे', 2.5), 
        ('भाजीपाला', 2.5), ('धान्य', 2.5), ('गहू', 2.5), ('तांदूळ', 2.5), ('ज्वारी', 2.5), 
        ('बाजरी', 2.5), ('मका', 2.5), ('कांदा', 2.5), ('बटाटा', 2.5), ('टोमॅटो', 2.5), 
        ('कापूस', 2.5), ('ऊस', 2.5), ('सोयाबीन', 2.5), ('तूर', 2.5), ('मूग', 2.5), 
        ('जमीन', 2.2), ('मशागत', 2.5), ('खते', 2.5), ('कीटकनाशके', 2.5), ('बियाणे', 2.5), 
        ('पाणी', 2.2), ('सिंचन', 2.5), ('पाऊस', 2.2), ('जलसिंचन', 2.5), ('विहीर', 2.2), 
        ('बोरवेल', 2.2), ('ट्रॅक्टर', 2.5), ('नांगर', 2.2), ('फवारणी', 2.2), ('कापणी', 2.2), 
        ('बाजारभाव', 2.5), ('मंडी', 2.5), ('कृषी उत्पन्न बाजार समिती', 2.8), 
        ('एपीएमसी', 2.5), ('आत्महत्या', 2.2), ('कर्जमाफी', 2.5), ('पीकविमा', 2.5), 
        ('दुष्काळ', 2.5), ('अवकाळी पाऊस', 2.5), ('गारपीट', 2.5)
    ]
}

# Improved weights for different categories
CATEGORY_WEIGHTS = {
    'Technology': 1.3,
    'Entertainment': 1.3,
    'Sports': 1.6,    # Sports terms are very specific
    'Politics': 1.4,  # Political terms may be frequent in news
    'Business': 1.2,
    'Health': 1.4,    # Health terms are quite specific
    'Education': 1.2,
    'Environment': 1.3,
    'Crime': 1.5,     # Crime terms are very specific
    'Agriculture': 1.4 # Agriculture terms are specific
}

# Common Marathi place names
COMMON_PLACES = [
    'मुंबई', 'पुणे', 'नागपूर', 'औरंगाबाद', 'नाशिक', 'सोलापूर', 'कोल्हापूर', 'अमरावती',
    'ठाणे', 'अकोला', 'जळगाव', 'अहमदनगर', 'सांगली', 'सातारा', 'रत्नागिरी', 'धुळे',
    'परभणी', 'बीड', 'लातूर', 'उस्मानाबाद', 'हिंगोली', 'बुलढाणा', 'यवतमाळ', 'वर्धा',
    'गोंदिया', 'भंडारा', 'चंद्रपूर', 'गडचिरोली', 'वाशिम', 'नंदुरबार', 'जालना',
    'सिंधुदुर्ग', 'रायगड', 'महाराष्ट्र', 'विदर्भ', 'मराठवाडा', 'कोकण', 'पश्चिम महाराष्ट्र'
]

def extract_places(text):
    """
    Extract place names from text.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of place names found in the text
    """
    places = []
    for place in COMMON_PLACES:
        if place in text:
            places.append(place)
    
    # Also look for places at the beginning of sentences (common in news articles)
    # Format is typically "Place: Rest of the news"
    place_pattern = r'^([^:]+):'
    matches = re.findall(place_pattern, text)
    for match in matches:
        if match.strip() not in places:
            places.append(match.strip())
    
    return places

def extract_emotions(text):
    """
    Extract emotions from text based on keywords with weighted importance.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with counts of positive, negative, and neutral emotions
    """
    emotion_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for keyword, weight in keywords:
            if keyword in text:
                emotion_counts[emotion] += weight
    
    # Determine the dominant emotion
    if sum(emotion_counts.values()) == 0:
        dominant_emotion = 'neutral'
    else:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    return {
        'counts': emotion_counts,
        'dominant': dominant_emotion
    }

def extract_severity(text):
    """
    Extract severity level from text based on keywords with weighted importance.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Severity level ('high', 'medium', 'low')
    """
    severity_counts = {'high': 0, 'medium': 0, 'low': 0}
    
    for level, keywords in SEVERITY_KEYWORDS.items():
        for keyword, weight in keywords:
            if keyword in text:
                severity_counts[level] += weight
    
    # Determine the dominant severity level
    if sum(severity_counts.values()) == 0:
        return 'low'  # Default to low if no severity keywords found
    else:
        return max(severity_counts, key=severity_counts.get)

def extract_category(text):
    """
    Extract category from text based on keywords with improved accuracy.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Category ('Technology', 'Entertainment', 'Sports', etc.)
    """
    # Initialize weighted scores for each category
    category_scores = {category: 0 for category in CATEGORY_KEYWORDS.keys()}
    
    # Count keyword occurrences with position bias (words at the beginning are more important)
    words = text.split()
    total_words = len(words)
    
    # Check for keywords in the text
    for category, keywords in CATEGORY_KEYWORDS.items():
        weight = CATEGORY_WEIGHTS.get(category, 1.0)
        
        for keyword, keyword_weight in keywords:
            if keyword in text:
                # Basic occurrence
                category_scores[category] += keyword_weight * weight
                
                # Check if keyword appears in the first 20% of the text (title or intro)
                # This is especially important for news articles
                first_part = ' '.join(words[:max(1, int(total_words * 0.2))])
                if keyword in first_part:
                    category_scores[category] += 2 * keyword_weight * weight  # Extra weight for early appearance
    
    # Check for title patterns (e.g., "Sports: ..." or "Tech News: ...")
    title_indicators = {
        'Technology': ['तंत्रज्ञान', 'टेक', 'डिजिटल', 'मोबाईल', 'गॅजेट'],
        'Entertainment': ['मनोरंजन', 'सिनेमा', 'चित्रपट', 'संगीत', 'कला'],
        'Sports': ['क्रीडा', 'खेळ', 'क्रिकेट', 'फुटबॉल', 'स्पोर्ट्स'],
        'Politics': ['राजकारण', 'राजकीय', 'निवडणूक', 'सरकार'],
        'Business': ['व्यवसाय', 'अर्थ', 'बिझनेस', 'मार्केट', 'बाजार'],
        'Health': ['आरोग्य', 'वैद्यकीय', 'हेल्थ', 'मेडिकल'],
        'Education': ['शिक्षण', 'विद्या', 'शैक्षणिक'],
        'Environment': ['पर्यावरण', 'निसर्ग', 'हवामान'],
        'Crime': ['गुन्हा', 'अपराध', 'चोरी', 'खून'],
        'Agriculture': ['शेती', 'कृषी', 'पीक', 'फळे']
    }
    
    for category, indicators in title_indicators.items():
        for indicator in indicators:
            if text.startswith(indicator) or f"{indicator}:" in text[:50]:
                category_scores[category] += 5  # Strong indicator in title
    
    # Determine the dominant category
    if sum(category_scores.values()) == 0:
        return 'Other'  # Default to Other if no category keywords found
    else:
        return max(category_scores, key=category_scores.get)

def extract_entities(text):
    """
    Extract all entities from text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with extracted entities
    """
    places = extract_places(text)
    emotions = extract_emotions(text)
    severity = extract_severity(text)
    category = extract_category(text)
    
    return {
        'places': places,
        'emotions': emotions,
        'severity': severity,
        'category': category
    }

def analyze_corpus(texts):
    """
    Analyze a corpus of texts to extract entities.
    
    Args:
        texts (list): List of texts
        
    Returns:
        dict: Dictionary with analysis results
    """
    all_places = []
    all_emotions = {'positive': 0, 'negative': 0, 'neutral': 0}
    all_severity = {'high': 0, 'medium': 0, 'low': 0}
    all_categories = {category: 0 for category in CATEGORY_KEYWORDS.keys()}
    all_categories['Other'] = 0  # Add 'Other' category
    
    for text in texts:
        entities = extract_entities(text)
        
        # Collect places
        all_places.extend(entities['places'])
        
        # Collect emotions
        all_emotions[entities['emotions']['dominant']] += 1
        
        # Collect severity
        all_severity[entities['severity']] += 1
        
        # Collect categories
        all_categories[entities['category']] += 1
    
    # Count occurrences of places
    place_counts = Counter(all_places)
    
    return {
        'places': dict(place_counts),
        'emotions': all_emotions,
        'severity': all_severity,
        'categories': all_categories
    }

def cluster_by_entity(df, entity_type):
    """
    Cluster articles based on a specific entity type.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles
        entity_type (str): Entity type to cluster by ('places', 'emotions', 'severity', 'category')
        
    Returns:
        pandas.DataFrame: Dataframe with cluster labels
    """
    df_with_clusters = df.copy()
    
    if entity_type == 'places':
        # Extract places for each article
        df_with_clusters['places'] = df_with_clusters['text'].apply(extract_places)
        
        # Assign cluster based on the first place mentioned (if any)
        df_with_clusters['cluster'] = df_with_clusters['places'].apply(
            lambda x: x[0] if x else 'unknown'
        )
    
    elif entity_type == 'emotions':
        # Extract emotions for each article
        df_with_clusters['emotions'] = df_with_clusters['text'].apply(
            lambda x: extract_emotions(x)['dominant']
        )
        
        # Assign cluster based on dominant emotion
        df_with_clusters['cluster'] = df_with_clusters['emotions']
    
    elif entity_type == 'severity':
        # Extract severity for each article
        df_with_clusters['severity'] = df_with_clusters['text'].apply(extract_severity)
        
        # Assign cluster based on severity
        df_with_clusters['cluster'] = df_with_clusters['severity']
    
    elif entity_type == 'category':
        # Extract category for each article
        df_with_clusters['category'] = df_with_clusters['text'].apply(extract_category)
        
        # Assign cluster based on category
        df_with_clusters['cluster'] = df_with_clusters['category']
    
    else:
        raise ValueError(f"Unsupported entity type: {entity_type}")
    
    return df_with_clusters

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत.",
        "पुणे: पुण्यात मोठ्या प्रमाणात पाऊस झाल्याने अनेक भागात पूरस्थिती निर्माण झाली आहे.",
        "नागपूर: नागपूर विद्यापीठाने परीक्षांचे वेळापत्रक जाहीर केले आहे."
    ]
    
    # Extract entities from each text
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text}")
        entities = extract_entities(text)
        print(f"Places: {entities['places']}")
        print(f"Emotions: {entities['emotions']}")
        print(f"Severity: {entities['severity']}")
        print(f"Category: {entities['category']}")
    
    # Analyze the corpus
    print("\nCorpus Analysis:")
    analysis = analyze_corpus(sample_texts)
    print(f"Places: {analysis['places']}")
    print(f"Emotions: {analysis['emotions']}")
    print(f"Severity: {analysis['severity']}")
    print(f"Categories: {analysis['categories']}")
