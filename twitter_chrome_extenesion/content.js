console.log("Twitter Hate Speech Analyzer content script loaded.");

const GRADIO_API_URL = "http://127.0.0.1:7860/analyze_tweet_extension_api";

function createAnalyzeButton(tweetTextArea) {
    const button = document.createElement("button");
    button.textContent = "🔍 Analyze Tweet";
    button.className = "analyze-tweet-button"; //css
    button.style.marginLeft = "10px";
    button.style.padding = "8px 12px";
    button.style.backgroundColor = "#1DA1F2";
    button.style.color = "white";
    button.style.border = "none";
    button.style.borderRadius = "20px";
    button.style.cursor = "pointer";
    button.style.fontSize = "14px";

    button.addEventListener("click", async () => {
        const tweetText = tweetTextArea.value || (tweetTextArea.innerText && tweetTextArea.innerText.trim());
        if (!tweetText || tweetText.trim() === "") {
            displayResults("Please enter some text to analyze.", "", "", tweetTextArea);
            return;
        }

        button.textContent = "Analyzing...";
        button.disabled = true;

        try {
            const response = await fetch(GRADIO_API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    data: [tweetText],
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log("Analysis result:", result);

            if (result.data && result.data.length >= 3) {
                displayResults(result.data[0], result.data[1], result.data[2], tweetTextArea);
            } else {
                displayResults("Error: Unexpected response format from server.", "", "", tweetTextArea);
                console.error("Unexpected response:", result);
            }

        } catch (error) {
            console.error("Error analyzing tweet:", error);
            displayResults(`Error: Could not connect to analysis server or server error. ${error.message}`, "", "", tweetTextArea);
        } finally {
            button.textContent = "🔍 Analyze Tweet";
            button.disabled = false;
        }
    });
    return button;
}

// ... (createAnalyzeButton and other functions remain the same) ...

function displayResults(classification, raschScore, explanation, referenceElement) {
    let resultsDiv = document.getElementById("tweet-analysis-results");
    let isNewDiv = false;
    if (!resultsDiv) {
        isNewDiv = true;
        resultsDiv = document.createElement("div");
        resultsDiv.id = "tweet-analysis-results";
        resultsDiv.style.marginTop = "5px";
        // Padding: top, right (for icon), bottom, left.
        // 52px right padding = 10px (initial right padding for text) + 32px (icon width) + 10px (buffer between text and icon)
        resultsDiv.style.padding = "10px 10px 10px 10px";
        resultsDiv.style.border = "1px solid #ccc";
        resultsDiv.style.borderRadius = "8px";
        resultsDiv.style.backgroundColor = "#ffffff";
        // position: relative; is now in style.css

        if (referenceElement && referenceElement.parentElement) {
            const buttonContainer = referenceElement.closest('div[data-testid="toolBar"]');
            if (buttonContainer && buttonContainer.parentElement) {
                buttonContainer.parentElement.insertAdjacentElement('afterend', resultsDiv);
            } else if (referenceElement.parentElement.parentElement) {
                referenceElement.parentElement.parentElement.insertAdjacentElement('afterend', resultsDiv);
            } else {
                document.body.appendChild(resultsDiv); // Fallback
            }
        } else {
             document.body.appendChild(resultsDiv); // Fallback
        }
    }

    // Clear previous icon if any to prevent multiple icons when re-analyzing
    const existingIcon = resultsDiv.querySelector(".classification-icon");
    if (existingIcon) {
        existingIcon.remove();
    }

    resultsDiv.innerHTML = `
        <h4>Analysis Results:</h4>
        <p><strong>Classification:</strong> ${classification}</p>
        <p><strong>Rasch Score:</strong> ${raschScore}</p>
        <p><strong>Explanation:</strong> ${explanation}</p>
    `;

    // Determine icon based on classification
    let iconFilename = "";
    if (classification === "Non-Hate") {
        iconFilename = "yes.jpg";
    } else if (classification === "Implicit Hate") {
        iconFilename = "no.jpg"; // Ensure this filename matches your file
    } else if (classification === "Explicit Hate") {
        iconFilename = "noo.jpg"; // Ensure this filename matches your file (case-sensitive)
    }
    // Add more conditions if there are other classifications like "Error..."
    else if (classification.startsWith("Error:") || classification.startsWith("Please enter")) {
        iconFilename = ""; // No icon for errors or input prompts
    }

    if (iconFilename) {
        const iconImg = document.createElement("img");
        iconImg.src = chrome.runtime.getURL(iconFilename);
        iconImg.alt = classification + " icon";
        iconImg.className = "classification-icon"; // Styled by CSS
        resultsDiv.appendChild(iconImg); // Append the icon to the results div
    }
}


// Function to find the tweet composer and add the button
function addAnalysisButtonToComposer() {
    const tweetTextArea = document.querySelector('div[data-testid="tweetTextarea_0"] div[role="textbox"], div[data-testid^="tweetTextarea_"] div[role="textbox"]');
    const tweetDialogTextArea = document.querySelector('div.DraftEditor-root');
    let targetTextArea = null;
    const mainTweetComposer = document.querySelector('div[data-testid="tweetTextarea_0"]');
    if (mainTweetComposer) {
        targetTextArea = mainTweetComposer.querySelector('div[contenteditable="true"]');
    }

    // If no textbox found, try to find a reply composer textarea
    if (!targetTextArea) {
        const replyTweetComposer = document.querySelector('div[data-testid^="tweetTextarea_"][data-testid$="_label"]');
        if (replyTweetComposer) {
            targetTextArea = replyTweetComposer.parentElement.querySelector('div[contenteditable="true"]');
        }
    }
    
    // If nothing not found, try the more general "textbox" role (less specific)
    if(!targetTextArea) {
        const textboxes = document.querySelectorAll('div[role="textbox"][contenteditable="true"]');
        // just find the  "Tweet" button as plan z
        for (let tb of textboxes) {
            if (tb.closest('div[data-testid="tweetButton"], div[data-testid="tweetButtonInline"]')) {
                targetTextArea = tb;
                break;
            }
        }
    }


    if (targetTextArea && !targetTextArea.dataset.hasAnalyzeButton) {
        console.log("Tweet text area found:", targetTextArea);
        const button = createAnalyzeButton(targetTextArea);
        const toolbar = targetTextArea.closest('div[data-testid="primaryColumn"]')?.querySelector('div[data-testid="toolBar"]');

        if (toolbar) {
            console.log("Toolbar found, appending button there.");
            const mediaButtons = toolbar.querySelector('div[data-testid="media-buttons"]');
            if (mediaButtons && mediaButtons.parentElement) {
                 mediaButtons.parentElement.appendChild(button);
            } else {
                toolbar.appendChild(button); 
            }
        } else if (targetTextArea.parentElement) {
            console.log("Toolbar not found, appending button after text area's parent.");
            targetTextArea.parentElement.insertAdjacentElement("afterend", button);
        }

        targetTextArea.dataset.hasAnalyzeButton = "true";
        console.log("Analyze button added.");
    }
}

const observer = new MutationObserver((mutationsList, observer) => {
    for(const mutation of mutationsList) {
        if (mutation.type === 'childList' || mutation.type === 'subtree') {
            addAnalysisButtonToComposer();
            break; 
        }
    }
});

observer.observe(document.body, { childList: true, subtree: true });
addAnalysisButtonToComposer();