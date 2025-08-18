let slideCount = 0;
let modifiedSlides = [];

window.addEventListener("DOMContentLoaded", () => {

    // Add slide
    function addSlide(data = {}) {
        const slideContainer = document.getElementById('slidesContainer');

        const slideDiv = document.createElement('div');
        slideDiv.classList.add('slide-block');
        slideDiv.innerHTML = `
            <h4>Slide ${slideCount + 1}</h4>

            <label>Slide Text:</label><br>
            <textarea name="slide_text_${slideCount}" required>${data.text || ""}</textarea><br><br>

            <label>Image:</label><br>
            <input type="file" name="slide_image_${slideCount}" accept="image/*" ${data.isLoaded ? "" : "required"}><br><br>

            <label>Audio (optional):</label><br>
            <input type="file" name="slide_audio_${slideCount}" accept="audio/*"><br><br>

            <label>Video (optional):</label><br>
            <input type="file" name="slide_video_${slideCount}" accept="video/*"><br><br>

            <label>Timestamps JSON (optional):</label><br>
            <input type="file" name="slide_json_${slideCount}" accept=".json"><br><br>

            <button type="button" class="remove-slide-btn">Remove Slide</button>
            <hr>
        `;

        slideDiv.querySelector('.remove-slide-btn').addEventListener('click', () => {
            slideDiv.remove();
            modifiedSlides = modifiedSlides.filter(slide => slide.id !== slideCount);
            slideCount--;
            updateSlideCount();
        });

        slideContainer.appendChild(slideDiv);
        slideCount++;
        updateSlideCount();

        // Track modified slides for later submission
        modifiedSlides.push({
            id: slideCount,
            text: data.text || "",
            image: null,
            audio: null,
            video: null,
            timestamps: null
        });
    }

    // Update the slide count field
    function updateSlideCount() {
        document.getElementById('slideCount').value = slideCount;
    }

    // Add Slide Button
    const addSlideBtn = document.getElementById('addSlideBtn');
    if (addSlideBtn) {
        addSlideBtn.addEventListener('click', function (e) {
            e.preventDefault();
            addSlide();
        });
    }

    // Upload Button
    const uploadButton = document.getElementById("uploadButton");
    if (uploadButton) {
        uploadButton.addEventListener("click", () => {
            document.getElementById("storyForm").style.display = "block";
            document.getElementById("loadForm").style.display = "none";
            document.getElementById("yourStoriesSection").style.display = "none";
        });
    }

    // Modify Button
    const modifyButton = document.getElementById("modifyButton");
    if (modifyButton) {
        modifyButton.addEventListener("click", () => {
            document.getElementById("loadForm").style.display = "block";
            document.getElementById("storyForm").style.display = "none";
            document.getElementById("yourStoriesSection").style.display = "none";
        });
    }

    // Your Stories Button
    const yourStoriesButton = document.getElementById("yourStoriesButton");
    if (yourStoriesButton) {
        yourStoriesButton.addEventListener("click", async () => {
            try {
                const response = await fetch("/api/my_stories", {
                    credentials: "same-origin"
                });
                if (!response.ok) throw new Error("Failed to fetch stories");

                const stories = await response.json();
                const list = document.getElementById("yourStoriesList");
                list.innerHTML = "";

                stories.forEach(story => {
                    const div = document.createElement("div");
                    div.classList.add("story-entry");

                    div.innerHTML = `
                        <strong>${story.title}</strong>
                        <button class="modify-btn" data-id="${story.id}">Modify</button>
                    `;

                    list.appendChild(div);
                });

                document.getElementById("yourStoriesSection").style.display = "block";

                document.querySelectorAll(".modify-btn").forEach(btn => {
                    btn.addEventListener("click", async (e) => {
                        const storyId = e.target.getAttribute("data-id");
                        const res = await fetch(`/api/story_data/${storyId}`, {
                            credentials: "same-origin"
                        });
                        const data = await res.json();

                        document.getElementById("title").value = data.title || "";
                        slideCount = 0;
                        document.getElementById("slidesContainer").innerHTML = "";
                        
                        // Load slides from the database (including timestamps)
                        (data.slides || []).forEach(slide => {
                            addSlide({ text: slide.text || "", isLoaded: true });
                            modifiedSlides[slideCount - 1].timestamps = slide.timestamps || [];
                        });

                        document.getElementById("storyForm").style.display = "block";
                        document.getElementById("loadForm").style.display = "none";
                        document.getElementById("yourStoriesSection").style.display = "none";
                    });
                });

            } catch (err) {
                alert("Error loading your stories.");
                console.error(err);
            }
        });
    }

    // Handle file changes for slide inputs
    function handleFileChange(e, slideId, type) {
        const file = e.target.files[0];
        if (!file) return;

        // Track modified slide files
        modifiedSlides.forEach(slide => {
            if (slide.id === slideId) {
                slide[type] = file; // Update the relevant field (image, audio, video)
            }
        });
    }

    // Listen for changes in file inputs (image, audio, video)
    document.addEventListener('change', function (e) {
        if (e.target.matches('input[type="file"]')) {
            const slideId = parseInt(e.target.name.split('_')[1]);
            const type = e.target.name.split('_')[0]; // image, audio, video
            handleFileChange(e, slideId, type);
        }
    });

    // Submit the modified data
    const submitButton = document.getElementById("submitButton");
    if (submitButton) {
        submitButton.addEventListener("click", async () => {
            const updatedSlides = modifiedSlides.map(slide => {
                return {
                    id: slide.id,
                    text: slide.text,
                    image: slide.image,
                    audio: slide.audio,
                    video: slide.video,
                    timestamps: slide.timestamps // Include timestamps from the modified slides
                };
            });

            try {
                const response = await fetch("/api/update_story", {
                    method: "POST",
                    body: JSON.stringify({
                        slides: updatedSlides,
                        // Include other data like title, user_id, etc.
                    }),
                    headers: {
                        "Content-Type": "application/json",
                    },
                });

                if (response.ok) {
                    alert("Story updated successfully!");
                } else {
                    alert("Failed to update the story.");
                }
            } catch (err) {
                console.error("Error submitting modified story:", err);
                alert("Error submitting the modified story.");
            }
        });
    }
});
