document.addEventListener('DOMContentLoaded', () => {
    const promptInput = document.getElementById('prompt-input');
    const submitBtn = document.getElementById('submit-btn');
    const resultsContainer = document.getElementById('results-container');

    const modelIdMap = {
        'DeepSeek': 'result-deepseek',
        'Qwen': 'result-qwen',
        'OpenAI': 'result-openai',
        'Doubao': 'result-doubao'
    };

    const API_URL = 'http://127.0.0.1:8000/api/query';

    submitBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('请输入问题！');
            return;
        }

        // --- 1. Set loading state --- 
        submitBtn.disabled = true;
        submitBtn.textContent = '查询中...';
        Object.values(modelIdMap).forEach(id => {
            const card = document.getElementById(id);
            const contentDiv = card.querySelector('.response-content');
            contentDiv.textContent = ''; // Clear previous content
            contentDiv.classList.add('loading');
        });

        try {
            // --- 2. Fetch data from backend --- 
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: prompt }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();

            // --- 3. Display results --- 
            results.forEach(result => {
                const cardId = modelIdMap[result.model];
                if (cardId) {
                    const card = document.getElementById(cardId);
                    const contentDiv = card.querySelector('.response-content');
                    contentDiv.classList.remove('loading');
                    if (result.error) {
                        contentDiv.textContent = `错误: ${result.error}`;
                    } else {
                        contentDiv.textContent = result.response;
                    }
                }
            });

        } catch (error) {
            // --- Handle fetch or network errors --- 
            console.error('Fetch error:', error);
            alert(`请求失败: ${error.message}`);
            // Clear loading state on error
            Object.values(modelIdMap).forEach(id => {
                const card = document.getElementById(id);
                card.querySelector('.response-content').classList.remove('loading');
            });
        } finally {
            // --- 4. Restore button state --- 
            submitBtn.disabled = false;
            submitBtn.textContent = '提交查询';
        }
    });
});

