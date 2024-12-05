def prompting():
    PROMPT = {
        'accounting': """When creating question and answer pairs, follow these steps:
1.Read the given notes and extract general keywords or topics.
2.Create new questions based on the extracted keywords or topics. The questions should be fact-based and require in-depth reasoning to answer.
3.Create possible answer options for the question, ensuring that only one answer is correct.
4.Provide a reason explaining the correct answer.
5.Then, return the question, answer options, correct answer, and reason in the format below.

Please maintain the following format for responses:
Question: (Your generated question)
Answer options: (Your generated answer options, with a minimum of 4 and a maximum of 8 options)
Answer: (The correct answer from the options)
Reason: (Explanation about answer)

Below is the content of a given lecture notes:
content: {document}
However, you must create questions that can be solved even without the given content. Derive a general topic from the article and create questions that require in-depth reasoning to answer.

Keeping the task in mind, provide question-and-answer pairs based on the given document.
Question:
Answer options:
Answer:
Reason:""",

        'book': """When creating question and answer pairs, follow these steps:
1.Read the given documents and extract general keywords or topics.
2.Create new questions based on the extracted keywords or topics. The questions should be fact-based and require in-depth reasoning to answer.
3.Create possible answer options for the question, ensuring that only one answer is correct.
4.Provide a reason explaining the correct answer.
5.Then, return the question, answer options, correct answer, and reason in the format below.

Please maintain the following format for responses:
Question: (Your generated question)
Answer options: (Your generated answer options, with a minimum of 4 and a maximum of 8 options)
Answer: (The correct answer from the options)
Reason: (Explanation about answer)

Below is the content of a given documents:
content: {document}
However, you must create questions that can be solved even without the given news content. Derive a general topic from the article and create questions that require in-depth reasoning to answer.

Keeping the task in mind, provide question-and-answer pairs based on the given news article.
Question:
Answer options:
Answer:
Reason:""",

        'account_qa': """When creating accounting problems and answers, follow these steps:
1.Review various accounting problems provided and extract general concepts or principles.
2.Create new accounting problems based on the extracted concepts or principles. The problems should be similar but not identical to the given ones and require in-depth reasoning to solve.
3.Generate multiple-choice answer options for the problem, ensuring that only one answer is correct.
4.Provide a detailed solution explaining how to arrive at the correct answer.
5.Then, return the problem, answer options, correct answer, and solution in the format below.

Please maintain the following format for responses:
Question: (Your generated question)
Answer options: (Your generated answer options, with a minimum of 4 and a maximum of 8 options)
Answer: (The correct answer from the options)
Solution: (Explanation about answer)

Below is the example of accounting problems:
Example1. q: {question1} 
Example2. q: {question2} 
Example3. q: {question3} 
However, you must create accounting problems that can be solved even without the given examples. Derive a general concept or principle from the provided problems and create new questions that require in-depth reasoning to solve.

Keeping the task in mind, provide question-and-answer pairs.
Question:
Answer options:
Answer:
Solution:""",
    }

    return PROMPT