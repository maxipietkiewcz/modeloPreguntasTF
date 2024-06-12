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
      "what is javascript?",
      "Explain the difference between let and var.",
      "What is the purpose of the `this` keyword?",
    ];

    const predefinedAnswers = [
      [
        "My dog is black",
        "A closure is a variable that is not used outside of its function.",
        "A closure is a function having access to the parent scope, even after the parent function has closed.",
      ],
      [
        "The car is blue",
        "var is function scoped and is raised initializing as undefined, allowing re-declaration; let is block-scoped, is raised without initialization, and does not allow re-declaration.",
        "let is block-scoped, while var is function-scoped. let cannot be redeclared in the same scope.",
      ],
      [
        "I love JavaScript",
        "The this keyword refers to the object from which a function is invoked. Its value depends on the context in which the function is called, allowing access to properties and methods of the object in question.",
        "JavaScript is a programming language used to build interactive websites.",
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
