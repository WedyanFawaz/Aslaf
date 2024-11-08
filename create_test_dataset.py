from langchain_community.document_loaders import DirectoryLoader # read docs from dir
from langchain.text_splitter import RecursiveCharacterTextSplitter # splitting text into chunks
from langchain.schema import Document # to use document class such that each chunk is doc with metadeta and content
from langchain_community.vectorstores import Chroma # chromadb to generate vector db


from langchain.embeddings import SentenceTransformerEmbeddings # to embed text
from langchain_openai import OpenAIEmbeddings #
import openai
import os # to prompt the system
import shutil # deleting and adding dirs

# ibm watsonx dep
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model


from datasets import Dataset # to create test dataset


def format_docs(docs : list[Document]):
    """
    Taked a lsit of docs and extract page content.
    """
    return "\n\n".join(doc.page_content for doc in docs) # turn docs into an easy to read formatted text

def create_dataset(retriever, generator):
    questions = [
    "ماذا علم الله لآدم في قوله سبحانه: {وَعَلَّمَ آدَمَ الْأَسْمَاءَ كُلَّهَا}؟",
    "كيف نشأت اللغات المختلفة حسب النص؟",
    "ما هو الإعراب؟",
    "ما هي اللغة؟",
    "كيف يتم الحكم على الكلمات عندما لا يوجد دليل أو نظير؟",
    "ما هما الضربان الذين تُقسم إليهما مقاييس العربية؟",
    "ما هي الأسباب المانعة من الصرف، وكيف تم تقسيمها؟",
    "كيف تم التعامل مع النصب في الجمع والتثنية؟",
    "كيف يرتبط القياس اللفظي بالمعنى؟",
    "كيف يتم حذف الجملة في اللغة العربية؟",
    "ماذا كان رأي أبو العباس في عدد حروف المعجم، ولماذا؟",
    "كيف تعبر الألف في النص عن وجودها في الكلمات وما هو دلالة ذلك؟"
]

    ground_truths = [
        "علم آدم أسماء جميع المخلوقات بجميع اللغات: العربية والفارسية والسريانية والعبرية والرومية وغير ذلك من سائر اللغات.",
        "ادم ولده تفرقوا في الدنيا وعلق كل منهم بلغة من تلك اللغات فغلبت عليه واضمحل عنه ما سواها لبعد عهدهم بها.",
        "هو الإبانة عن المعاني بالألفاظ ألا ترى أنك إذا سمعت أكرم سعيد أباه وشكر سعيدًا أبوه علمت برفع أحدهما ونصب الآخر الفاعل من المفعول ولو كان الكلام شرجًا واحدًا لاستبهم أحدهما من صاحبه. فأما لفظه فإنه مصدر أعربت عن الشيء إذا أوضحت عنه؛ وفلان معرب عما في نفسه أي مبين له، وموضح عنه. وأصل هذا كله قولهم 'العرب ' وذلك لما يعزى إليها من الفصاحة، والإعراب والبيان",
        "أما حدها 'فإنها أصوات' يعبر بها كل قوم عن أغراضهم. هذا حدها.وأما تصريفها ومعرفة حروفها فإنها فعلة من لغوت. أي تكلمت وأصلها لغوة ككرة وقلة وثبة كلها لاماتها واوات لقولهم.",
        "وأما إن لم يقم الدليل ولم يوجد النظير فإنك تحكم مع عدم النظير. وذلك كقولك في الهمزة والنون من أندلس: إنهما زائدتان، وإن وزن الكلمة بهما 'أنفعُل' وإن كان مثالا لا نظير له. وذلك أن النون لا محالة زائدة؛ لأنه ليس في ذوات الخمسة شيء على 'فعلَلُل' فتكون النون فيه أصلا لوقوعها موقع العين.",
        "وهي ضربان: أحدهما معنوي والآخر لفظي. وهذان الضربان وإن عمّا وفشوا في هذه اللغة فإن أقواهما وأوسعهما هو القياس المعنوي.",
        "ألا ترى أن الأسباب المانعة من الصرف تسعة: واحد منها لفظي وهو شبه الفعل لفظًا، نحو أحمد ويرمع وتنضب وإثمد وأبلم وبقم وإستبرق والثمانية الباقية كلها معنوية كالتعريف والوصف والعدل والتأنيث وغير ذلك.",
        "ألا ترى أنهم لما أعربوا بالحروف في التثنية والجمع الذي على حده فأعطوا الرفع في التثنية الألف والرفع في الجمع الواو والجر فيهما الياء وبقي النصب لا حرف له فيماز به، جذبوه إلى الجر فحملوه عليه دون الرفع لتلك الأسباب المعروفة هناك فلا حاجة بنا هنا إلى الإطالة بذكرها.",
        "واعلم أن القياس اللفظي إذا تأملته لم تجده عاريًا من اشتمال المعنى عليه؛ ألا ترى أنك إذا سئلت عن 'إن' من قوله: ورج الفتى للخير ما إن رأيته … على السن خيرًا لا يزال يزيد فإنك قائل: دخلت على 'ما' -وإن كانت 'ما' ههنا مصدرية- لشبهها لفظًا بما النافية التي تؤكد بإن من قوله: ما إن يكاد يخليهم لوجهتهم … تخالج الأمر إن الأمر مشترك.",
        "فأمَّا الجملة فنحو قولهم في القسم: والله لا فعلت، وتالله لقد فعلت. وأصله: أقسم بالله، فحذف الفعل والفاعل وبقيت الحال -من الجار والجواب- دليلًا على الجملة المحذوفة.",
        "إنه كان يعدها ثمانية وعشرين حرفا، ويجعل أولها الباء، ويدع الألف من أولها، ويقول: هي همزة، ولا تثبت على صورة واحدة، وليست لها صورة مستقرة، فلا أعتدها مع الحروف التي أشكالها محفوظة معروفة.",
        "اعلم أن الألف التي في أول حروف المعجم هي صورة الهمزة، وإنما كتبت الهمزة واوا مرة وياء أخرى على مذهب أهل الحجاز في التخفيف."
    ]
    answers = []
    contexts = []

    # Inference
    for query in questions:
        context = format_docs(retriever.get_relevant_documents(query))
        answer = generator.generate(query, context)['results'][0]['generated_text']
        answers.append(answer)
        contexts.append([doc.page_content for doc in retriever.get_relevant_documents(query)])

    # To dict
    data = {
        "question": questions,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    # Convert dict to dataset
    dataset = Dataset.from_dict(data)
    return dataset
