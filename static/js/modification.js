document.addEventListener("DOMContentLoaded", function () {
    // Handle form submission for saving slide changes
    const form = document.getElementById('modificationForm');
    if (form) {
        form.addEventListener('submit', function (e) {
            e.preventDefault();
            saveAllSlideChanges();
        });
    }

    // Handle Modify button clicks (only shown when listing stories)
    const modifyButtons = document.querySelectorAll(".modify-button");
    modifyButtons.forEach(button => {
        button.addEventListener("click", function () {
            const url = button.getAttribute("data-url");
            console.log("Redirecting to: ", url);  // For debugging
            if (url) {
                window.location.href = url;
            } else {
                console.error("No URL found for Modify button.");
            }
        });
    });
});

function saveAllSlideChanges() {
    const form = document.getElementById('modificationForm');
    if (!form) {
        alert("Form not found.");
        return;
    }

    const formData = new FormData();
    const slides = form.querySelectorAll('.slide-card');
    const slideIds = [];

    slides.forEach((slide) => {
        const slideId = slide.dataset.slideId;
        slideIds.push(slideId);

        const textarea = slide.querySelector(`textarea[name="text_${slideId}"]`);
        if (textarea) {
            formData.append(`text_${slideId}`, textarea.value);
        }

        const imageInput = slide.querySelector(`input[name="image_${slideId}"]`);
        if (imageInput && imageInput.files.length > 0) {
            formData.append(`image_${slideId}`, imageInput.files[0]);
        }

        const videoInput = slide.querySelector(`input[name="video_${slideId}"]`);
        if (videoInput && videoInput.files.length > 0) {
            formData.append(`video_${slideId}`, videoInput.files[0]);
        }
    });

    formData.append('slide_ids', JSON.stringify(slideIds));

    const saveBtn = form.querySelector('.save-btn');
    if (saveBtn) saveBtn.disabled = true;

    fetch('/api/update_slides', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Slides updated successfully!');
                window.location.reload();
            } else {
                alert('Failed to update slides.');
            }
        })
        .catch(error => {
            console.error('Update error:', error);
            alert('Something went wrong.');
        })
        .finally(() => {
            if (saveBtn) saveBtn.disabled = false;
        });
}
