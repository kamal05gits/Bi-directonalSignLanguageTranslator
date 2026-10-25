(() => {
    const input = document.getElementById('text-to-sign-input');
    const form = document.getElementById('text-to-sign-form');
    if (!input || !form) return;

    const languageSelect = document.getElementById('text-sign-language');
    const avatarSelect = document.getElementById('text-sign-avatar');
    const englishPreview = document.getElementById('text-sign-english');
    const listenButton = document.getElementById('text-sign-listen-btn');
    const catalogs = window.signoraSignCatalog && typeof window.signoraSignCatalog === 'object'
        ? window.signoraSignCatalog
        : {};
    let catalog = [];
    let signs = new Map();
    const elements = {
        characterCount: document.getElementById('character-count'),
        clear: document.getElementById('clear-text-btn'),
        detail: document.getElementById('sign-step-detail'),
        frameVideo: document.getElementById('sign-frame-video'),
        frameText: document.getElementById('sign-frame-text'),
        loop: document.getElementById('loop-signs'),
        next: document.getElementById('next-sign-btn'),
        play: document.getElementById('play-signs-btn'),
        placeholder: document.getElementById('sign-placeholder'),
        previous: document.getElementById('previous-sign-btn'),
        progress: document.querySelector('.sign-progress'),
        progressBar: document.getElementById('sign-progress-bar'),
        sequence: document.getElementById('sign-sequence'),
        sequenceCount: document.getElementById('sequence-count'),
        speed: document.getElementById('playback-speed'),
        status: document.getElementById('text-sign-status'),
        stepCounter: document.getElementById('sign-step-counter'),
        title: document.getElementById('sign-step-title')
    };

    const phraseAliases = [
        { words: ['THANK', 'YOU'], sign: 'THANKYOU' },
        { words: ['GOOD', 'BYE'], sign: 'GOODBYE' }
    ];
    let steps = [];
    let currentIndex = -1;
    let playbackTimer = null;
    let playing = false;
    let recognition = null;
    let translationRequestId = 0;
    let automaticUpdateTimer = null;

    function selectAvatarCatalog() {
        catalog = Array.isArray(catalogs[avatarSelect.value]) ? catalogs[avatarSelect.value] : [];
        signs = new Map(catalog.map(sign => [normalizeWord(sign.name), sign]));
    }

    function normalizeWord(value) {
        return String(value)
            .normalize('NFD')
            .replace(/[\u0300-\u036f]/g, '')
            .toUpperCase()
            .replace(/[^A-Z0-9]/g, '');
    }

    function setStatus(message, isError = false) {
        elements.status.textContent = message;
        elements.status.classList.toggle('is-error', isError);
    }

    function setPlaying(value) {
        playing = value;
        elements.play.innerHTML = playing
            ? '<i class="fas fa-pause me-2"></i>Pause'
            : '<i class="fas fa-play me-2"></i>Play';
        elements.play.setAttribute('aria-label', playing ? 'Pause playback' : 'Play signs');
    }

    function stopPlayback() {
        if (playbackTimer !== null) {
            window.clearTimeout(playbackTimer);
            playbackTimer = null;
        }
        elements.frameVideo.pause();
        setPlaying(false);
    }

    function appendWord(word, wordIndex, tokens) {
        for (const alias of phraseAliases) {
            const matchingWords = tokens.slice(wordIndex, wordIndex + alias.words.length);
            if (matchingWords.length !== alias.words.length
                || !matchingWords.every((token, index) => normalizeWord(token) === alias.words[index])) {
                continue;
            }
            const sign = signs.get(alias.sign);
            if (sign) {
                steps.push({
                    detail: 'Word sign',
                    label: alias.words.join(' '),
                    sign,
                    type: 'word'
                });
                return alias.words.length;
            }
        }

        const normalized = normalizeWord(word);
        const sign = signs.get(normalized);
        if (sign) {
            steps.push({ detail: 'Word sign', label: word, sign, type: 'word' });
            return 1;
        }

        const spellable = Array.from(normalized).filter(character => signs.has(character));
        if (spellable.length) {
            spellable.forEach(character => {
                steps.push({
                    detail: `Finger-spelling “${word}”`,
                    label: character,
                    sign: signs.get(character),
                    type: 'letter'
                });
            });
        } else {
            steps.push({ detail: 'No matching sign reference', label: word, type: 'unsupported' });
        }
        return 1;
    }

    function createSequence(text) {
        const tokens = text.match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*|[.,!?;:]/gu) || [];
        steps = [];

        for (let index = 0; index < tokens.length;) {
            const token = tokens[index];
            if (/^[.,!?;:]$/.test(token)) {
                steps.push({ detail: 'Brief pause', label: token, type: 'pause' });
                index += 1;
            } else {
                index += appendWord(token, index, tokens);
            }
        }
    }

    async function translateToEnglish(text) {
        if (languageSelect.value === 'en') return text;

        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, lang: 'en' })
        });
        const result = await response.json();
        if (!response.ok || !result.success) {
            throw new Error(result.error || 'Could not translate this phrase into English.');
        }
        return result.translated;
    }

    async function showSignsForPhrase() {
        window.clearTimeout(automaticUpdateTimer);
        stopPlayback();
        const originalText = input.value.trim();
        if (!originalText) {
            setStatus('Enter or speak a phrase to create a sign sequence.', true);
            input.focus();
            return;
        }

        const requestId = ++translationRequestId;
        invalidateSequence();
        setStatus(languageSelect.value === 'en' ? 'Preparing sign sequence...' : 'Translating phrase to English...');
        try {
            const englishText = (await translateToEnglish(originalText)).trim();
            if (requestId !== translationRequestId) return;
            if (!englishText) {
                throw new Error('The English translation was empty.');
            }
            englishPreview.value = englishText;
            createSequence(englishText);
            renderSequence();
            if (!steps.length) {
                setStatus('No words, numbers, or punctuation could be read from the English phrase.', true);
                return;
            }

            const unsupportedCount = steps.filter(step => step.type === 'unsupported').length;
            const translatedNote = languageSelect.value === 'en' ? '' : ' English translation ready.';
            setStatus(unsupportedCount
                ? `Sequence ready.${translatedNote} ${unsupportedCount} item${unsupportedCount === 1 ? '' : 's'} could not be finger-spelled with the available references.`
                : `Sequence ready.${translatedNote} Select Play or choose any step to review it.`);
        } catch (error) {
            if (requestId !== translationRequestId) return;
            englishPreview.value = '';
            setStatus(error.message, true);
        }
    }

    function scheduleAutomaticUpdate() {
        window.clearTimeout(automaticUpdateTimer);
        translationRequestId += 1;
        englishPreview.value = '';
        invalidateSequence();

        if (!input.value.trim()) {
            setStatus('');
            return;
        }

        setStatus('Updating translation and signs...');
        automaticUpdateTimer = window.setTimeout(() => {
            showSignsForPhrase();
        }, 450);
    }

    function initializeSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            listenButton.disabled = true;
            setStatus('Speech input is not supported by this browser. You can type your phrase instead.');
            return;
        }

        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.onresult = event => {
            const transcript = event.results?.[0]?.[0]?.transcript?.trim();
            if (!transcript) {
                setStatus('No phrase was recognized. Please try again.', true);
                return;
            }
            input.value = transcript;
            elements.characterCount.textContent = String(input.value.length);
            showSignsForPhrase();
        };
        recognition.onerror = event => {
            const messages = {
                'not-allowed': 'Microphone access was denied. Allow microphone access and try again.',
                'service-not-allowed': 'Speech recognition is unavailable for this site.',
                'no-speech': 'No speech was detected. Please try again.',
                'audio-capture': 'No microphone was found. Connect a microphone and try again.',
                'language-not-supported': 'Speech recognition does not support the selected language.',
                'network': 'Speech recognition could not connect to its service. Check the network and try again.'
            };
            setStatus(messages[event.error] || `Speech recognition failed: ${event.error}.`, true);
        };
        recognition.onstart = () => {
            setStatus(`Listening in ${languageSelect.selectedOptions[0].text}...`);
        };
        recognition.onend = () => {
            listenButton.disabled = false;
            listenButton.innerHTML = '<i class="fas fa-microphone me-2"></i>Speak phrase';
            if (elements.status.textContent.startsWith('Listening in ')) {
                setStatus('No phrase was recognized. Please try again.', true);
            }
        };
    }

    function renderSequence() {
        elements.sequence.replaceChildren();
        const fragment = document.createDocumentFragment();
        steps.forEach((step, index) => {
            const button = document.createElement('button');
            button.className = 'sign-sequence-step';
            button.type = 'button';
            button.dataset.index = String(index);
            button.setAttribute('aria-label', `Step ${index + 1}: ${step.label}, ${step.detail}`);

            const label = document.createElement('span');
            label.textContent = step.label;
            const detail = document.createElement('small');
            detail.textContent = step.type === 'letter' ? 'letter' : step.type;
            button.append(label, detail);
            button.addEventListener('click', () => {
                stopPlayback();
                showStep(index);
            });
            fragment.appendChild(button);
        });
        elements.sequence.appendChild(fragment);

        elements.sequenceCount.textContent = `${steps.length} ${steps.length === 1 ? 'step' : 'steps'}`;
        elements.play.disabled = steps.length === 0;
        elements.previous.disabled = steps.length === 0;
        elements.next.disabled = steps.length === 0;
        if (steps.length) showStep(0);
        else showEmptyState();
    }

    function invalidateSequence() {
        stopPlayback();
        steps = [];
        renderSequence();
    }

    function showEmptyState() {
        currentIndex = -1;
        elements.frameVideo.pause();
        elements.frameVideo.removeAttribute('src');
        elements.frameVideo.load();
        elements.frameVideo.hidden = true;
        elements.frameText.hidden = true;
        elements.placeholder.hidden = false;
        elements.title.textContent = 'Ready when you are';
        elements.detail.textContent = '';
        elements.stepCounter.textContent = '0 / 0';
        elements.progressBar.style.width = '0%';
        elements.progress.setAttribute('aria-valuenow', '0');
    }

    function showStep(index) {
        if (!steps.length) return;
        currentIndex = (index + steps.length) % steps.length;
        const step = steps[currentIndex];
        const progress = Math.round(((currentIndex + 1) / steps.length) * 100);
        elements.title.textContent = step.type === 'pause' ? 'Pause' : step.label;
        elements.detail.textContent = step.detail;
        elements.stepCounter.textContent = `${currentIndex + 1} / ${steps.length}`;
        elements.progressBar.style.width = `${progress}%`;
        elements.progress.setAttribute('aria-valuenow', String(progress));

        elements.frameVideo.pause();
        elements.frameVideo.hidden = true;
        elements.frameText.hidden = true;
        elements.placeholder.hidden = true;

        if (step.sign) {
            elements.frameVideo.src = step.sign.url;
            elements.frameVideo.setAttribute(
                'aria-label',
                `Animated ISL ${step.type === 'letter' ? 'finger-spelling letter' : 'word sign'}: ${step.label}`
            );
            elements.frameVideo.playbackRate = Number(elements.speed.value);
            elements.frameVideo.hidden = false;
        } else {
            elements.frameText.textContent = step.label;
            elements.frameText.hidden = false;
        }

        elements.sequence.querySelectorAll('.sign-sequence-step').forEach((button, buttonIndex) => {
            const active = buttonIndex === currentIndex;
            button.classList.toggle('active', active);
            if (active) button.setAttribute('aria-current', 'step');
            else button.removeAttribute('aria-current');
        });
        const activeStep = elements.sequence.querySelector('.sign-sequence-step.active');
        if (activeStep) activeStep.scrollIntoView({ block: 'nearest', inline: 'nearest' });

        if (playing) playCurrentStep();
    }

    function advancePlayback() {
        if (currentIndex === steps.length - 1) {
            if (elements.loop.checked) showStep(0);
            else stopPlayback();
            return;
        }
        showStep(currentIndex + 1);
    }

    function playCurrentStep() {
        const step = steps[currentIndex];
        if (step && step.sign) {
            elements.frameVideo.playbackRate = Number(elements.speed.value);
            elements.frameVideo.play().catch(error => {
                if (!playing || error.name === 'AbortError') return;
                stopPlayback();
                setStatus(`Could not play this sign video: ${error.message || 'playback was rejected.'}`, true);
            });
        } else {
            playbackTimer = window.setTimeout(() => {
                playbackTimer = null;
                advancePlayback();
            }, 1200 / Number(elements.speed.value));
        }
    }

    function scheduleNextStep() {
        if (playing) playCurrentStep();
    }

    function togglePlayback() {
        if (!steps.length) return;
        if (playing) {
            stopPlayback();
            return;
        }
        if (currentIndex >= steps.length - 1) showStep(0);
        setPlaying(true);
        scheduleNextStep();
    }

    form.addEventListener('submit', event => {
        event.preventDefault();
        showSignsForPhrase();
    });

    listenButton.addEventListener('click', () => {
        if (!recognition) return;
        recognition.lang = languageSelect.selectedOptions[0].dataset.locale;
        listenButton.disabled = true;
        listenButton.innerHTML = '<i class="fas fa-ear-listen me-2"></i>Listening...';
        setStatus(`Listening in ${languageSelect.selectedOptions[0].text}...`);
        try {
            recognition.start();
        } catch (error) {
            listenButton.disabled = false;
            listenButton.innerHTML = '<i class="fas fa-microphone me-2"></i>Speak phrase';
            setStatus(`Could not start speech recognition: ${error.message}`, true);
        }
    });

    elements.play.addEventListener('click', togglePlayback);
    elements.previous.addEventListener('click', () => {
        stopPlayback();
        showStep(currentIndex - 1);
    });
    elements.next.addEventListener('click', () => {
        stopPlayback();
        showStep(currentIndex + 1);
    });
    elements.frameVideo.addEventListener('ended', () => {
        if (playing) advancePlayback();
    });
    elements.frameVideo.addEventListener('error', () => {
        stopPlayback();
        setStatus('Could not load this sign video. Check that the clip is available.', true);
    });
    elements.speed.addEventListener('change', () => {
        elements.frameVideo.playbackRate = Number(elements.speed.value);
    });

    elements.clear.addEventListener('click', () => {
        window.clearTimeout(automaticUpdateTimer);
        stopPlayback();
        translationRequestId += 1;
        input.value = '';
        elements.characterCount.textContent = '0';
        englishPreview.value = '';
        steps = [];
        renderSequence();
        setStatus('');
        input.focus();
    });

    input.addEventListener('input', () => {
        elements.characterCount.textContent = String(input.value.length);
        scheduleAutomaticUpdate();
    });

    languageSelect.addEventListener('change', () => {
        scheduleAutomaticUpdate();
    });

    avatarSelect.addEventListener('change', () => {
        selectAvatarCatalog();
        scheduleAutomaticUpdate();
        if (!catalog.length) {
            setStatus(`No animated sign videos are available for ${avatarSelect.value}.`, true);
        }
    });

    document.querySelectorAll('[data-example]').forEach(button => {
        button.addEventListener('click', () => {
            languageSelect.value = 'en';
            input.value = button.dataset.example;
            elements.characterCount.textContent = String(input.value.length);
            form.requestSubmit();
        });
    });

    document.addEventListener('keydown', event => {
        const target = event.target;
        if (target === input || target instanceof HTMLSelectElement || target.isContentEditable) return;
        if (event.key === 'ArrowRight' && steps.length) {
            event.preventDefault();
            stopPlayback();
            showStep(currentIndex + 1);
        } else if (event.key === 'ArrowLeft' && steps.length) {
            event.preventDefault();
            stopPlayback();
            showStep(currentIndex - 1);
        } else if (event.code === 'Space' && steps.length && target instanceof HTMLElement
            && !['BUTTON', 'INPUT', 'TEXTAREA'].includes(target.tagName)) {
            event.preventDefault();
            togglePlayback();
        }
    });

    showEmptyState();
    initializeSpeechRecognition();
    selectAvatarCatalog();
    if (!catalog.length) {
        setStatus(`No animated sign videos are available for ${avatarSelect.value}.`, true);
    }
})();
