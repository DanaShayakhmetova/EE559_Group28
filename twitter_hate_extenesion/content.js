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
            displayResults("Please enter some text to analyze.", "", "");
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

function displayResults(classification, raschScore, explanation, referenceElement) {
    let resultsDiv = document.getElementById("tweet-analysis-results");
    if (!resultsDiv) {
        resultsDiv = document.createElement("div");
        resultsDiv.id = "tweet-analysis-results";
        resultsDiv.style.marginTop = "10px";
        resultsDiv.style.padding = "10px";
        resultsDiv.style.border = "1px solid #ccc";
        resultsDiv.style.borderRadius = "8px";
        resultsDiv.style.backgroundColor = "#f9f9f9";

        // Insert after the reference element's parent or a known container
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

    resultsDiv.innerHTML = `
        <h4>Analysis Results:</h4>
        <p><strong>Classification:</strong> ${classification}</p>
        <p><strong>Rasch Score:</strong> ${raschScore}</p>
        <p><strong>Explanation:</strong> ${explanation}</p>
    `;
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
            // Check if there's a specific spot, e.g., next to media buttons
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


        targetTextArea.dataset.hasAnalyzeButton = "true"; // Mark that button is added
        console.log("Analyze button added.");
    } else if (targetTextArea && targetTextArea.dataset.hasAnalyzeButton) {
        // console.log("Analyze button already exists for this text area."); this was debugging
    } else {
        // console.log("Tweet text area not found yet.");this was debugging
    }
}

// Twitter content is dynamic so every now and hten we check for tweet box.
// A MutationObserver is more efficient for this.
const observer = new MutationObserver((mutationsList, observer) => {
    for(const mutation of mutationsList) {
        if (mutation.type === 'childList' || mutation.type === 'subtree') {
            addAnalysisButtonToComposer();
            break;
        }
    }
});

// Start observing the document body for changes
observer.observe(document.body, { childList: true, subtree: true });

addAnalysisButtonToComposer();