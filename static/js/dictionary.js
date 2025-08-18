document.addEventListener("DOMContentLoaded", function () {
    const addButton = document.getElementById("add-entry");
    const container = document.getElementById("dictionary-entries");

    if (!addButton || !container) {
        console.error("Add button or dictionary container not found!");
        return;
    }

    addButton.addEventListener("click", function () {
        const newEntry = document.createElement("div");
        newEntry.classList.add("dictionary-entry");

        newEntry.innerHTML = `
            <label>Word:</label>
            <input type="text" name="word[]" placeholder="Enter word" required>

            <label>Image:</label>
            <input type="file" name="image[]" accept="image/*" required>

            <hr>
        `;

        container.appendChild(newEntry);
    });
});
