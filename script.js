import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as use from "@tensorflow-models/universal-sentence-encoder";
import * as tf from "@tensorflow/tfjs-core";

const initQnA = async () => {
  const model = await use.loadQnA();
  document.querySelector("#loadingQnA").style.display = "none";

  window.evaluateAnswers = async () => {
    const userAnswers = [
      document.getElementById("user_answer_1").value.trim(),
      document.getElementById("user_answer_2").value.trim(),
      document.getElementById("user_answer_3").value.trim(),
    ];

    const questions = [
      "What is a closure in JavaScript?",
      "Explain the difference between let and var.",
      "What is the purpose of the `this` keyword?",
    ];

    const predefinedAnswers = [
      [
        "My dog is black", // Wrong answer
        "A closure is a variable that is not used outside of its function.", // Acceptable answer
        "A closure is a function having access to the parent scope, even after the parent function has closed.", // Correct answer
      ],
      [
        "The car is blue", // Wrong answer
        "var is used to declare variables.", // Acceptable answer
        "let is block-scoped, while var is function-scoped. let cannot be redeclared in the same scope.", // Correct answer
      ],
      [
        "I love JavaScript", // Wrong answer
        "`this` refers to the current function.", // Acceptable answer
        "JavaScript is a programming language used to build interactive websites.", // Incorrect answer
      ],
    ];

    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    for (let i = 0; i < questions.length; i++) {
      const input = {
        queries: [questions[i]],
        responses: [userAnswers[i], ...predefinedAnswers[i]],
      };
      let result = await model.embed(input);
      const dp = tf
        .matMul(
          result["queryEmbedding"],
          result["responseEmbedding"],
          false,
          true
        )
        .dataSync();

      const resultDom = document.createElement("div");
      resultDom.innerHTML = `
        <h3>Results for Question ${i + 1}</h3>
        <p>Your Answer: ${userAnswers[i]} - Score: ${dp[0].toFixed(2)}</p>
        <p>Predefined Answer 1: ${
          predefinedAnswers[i][0]
        } - Score: ${dp[1].toFixed(2)}</p>
        <p>Predefined Answer 2: ${
          predefinedAnswers[i][1]
        } - Score: ${dp[2].toFixed(2)}</p>
        <p>Predefined Answer 3: ${
          predefinedAnswers[i][2]
        } - Score: ${dp[3].toFixed(2)}</p>
      `;

      resultsDiv.appendChild(resultDom);
    }
  };
};

initQnA();
