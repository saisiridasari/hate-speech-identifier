document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  const textarea = document.querySelector("textarea");

  form.addEventListener("submit", (e) => {
    if (textarea.value.trim() === "") {
      e.preventDefault();
      alert("Please enter a sentence before submitting.");
    }
  });
});
