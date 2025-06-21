# 🚀 Local Environment Setup (Using Cloud Resources)

Follow this guide to prepare your local machine for building Docker images and pushing them to Google Cloud.

---

## 1. Install Docker

Required for creating and pushing container images.

* Download and install Docker Desktop for your OS:
  [Official Docker installation guide](https://docs.docker.com/desktop/)

---

## 2. Install Z Shell (zsh)

We recommend using `zsh` for easier shell configuration and compatibility with cloud tooling.

**On Debian/Ubuntu:**

```bash
sudo apt update -y
sudo apt install zsh
```

For other platforms, choose the appropriate option here:
[zsh installation instructions](https://github.com/ohmyzsh/ohmyzsh/wiki/Installing-ZSH)

---

## 3. Install Oh‑My‑Zsh *(optional but helpful)*

Enhances `zsh` with themes, plugins, and better productivity.

```bash
# Using wget
sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Or using curl
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

Install any additional plugins you like for syntax highlighting, auto-suggestions, etc. I highly suggest the following two:

```bash
# zsh-autosuggestions – suggests commands based on history
git clone https://github.com/zsh-users/zsh-autosuggestions \
    ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# zsh-syntax-highlighting – colors commands to spot typos instantly
git clone https://github.com/zsh-users/zsh-syntax-highlighting \
    ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

After installing them, enable them in your `.zshrc`. Open your config:

```bash
nano ~/.zshrc
```

Find the `plugins=(...)` line and update:

```zsh
plugins=(
  git
  zsh-autosuggestions
  zsh-syntax-highlighting
)
```

These give you real-time history-based suggestions and visual syntax feedback ([hackernoon.com][1], [nevercodealone.medium.com][2], [ohmyz.sh][3], [github.com][4]).

* **Auto-suggestions**: As you type, commands from your history appear in gray—accept with → or `<Tab>` ([forum.manjaro.org][5], [linuxhandbook.com][6]).
* **Syntax highlighting**: Helps catch typos and see valid commands faster ([nevercodealone.medium.com][2]).

---

## 4. Install and Configure `mamba` (Conda alternative)

Fast environment manager with compatibility for Python dependencies.

```bash
wget -O Miniforge3.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3.sh -b -p "${HOME}/conda"
```

Then, add this to your `~/.zshrc` (or source it manually):

```zsh
# >>> mamba initialize >>>
export MAMBA_EXE='${HOME}/conda/bin/mamba'
export MAMBA_ROOT_PREFIX='${HOME}/conda'
__mamba_setup="$("$MAMBA_EXE" shell hook --shell zsh --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
if [ $? -eq 0 ]; then
  eval "$__mamba_setup"
else
  alias mamba="$MAMBA_EXE"
fi
unset __mamba_setup
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"
# <<< mamba initialize <<<
```

Reload your shell or run `source ~/.zshrc` to activate `mamba`.

---

## 5. Install Google Cloud SDK (`gcloud`)

Needed for authentication and pushing containers to your Google Cloud project.

1. Install following your OS instructions:
   [Google Cloud SDK install guide](https://cloud.google.com/sdk/docs/install)
2. Restart your shell to ensure `gcloud` is in `PATH`.
3. Authenticate your account:

   ```bash
   gcloud auth login
   ```
4. Enable Docker authentication for pushing images to Artifact Registry:

   ```bash
   gcloud auth configure-docker ${ARTIFACT_REGION}
   ```

   Replace `${ARTIFACT_REGION}` with your registry’s region (e.g., `us-central1`).

---

## ✅ You're All Set!

You now have:

* Docker locally for building and pushing images
* A modern shell environment with `zsh` and optional enhancements
* Python env management via `mamba`
* Authentication ready for Google Cloud operations

Next steps, found in notebooks:

1. Build your Docker image
2. Test it locally
3. Push it to your Artifact Registry for cloud deployment




[1]: https://hackernoon.com/customize-oh-my-zsh-with-syntax-highlighting-and-auto-suggestions-6q1b3w8o?utm_source=chatgpt.com "Customize Oh My Zsh with Syntax Highlighting and Auto-Suggestions"
[2]: https://nevercodealone.medium.com/oh-my-zsh-syntax-highlighting-plugin-c166f1400c4b?utm_source=chatgpt.com "oh-my-zsh syntax highlighting plugin | by Never Code Alone | Medium"
[3]: https://ohmyz.sh/?utm_source=chatgpt.com "Oh My Zsh - a delightful & open source framework for Zsh"
[4]: https://github.com/zsh-users/zsh-autosuggestions?utm_source=chatgpt.com "Fish-like autosuggestions for zsh - GitHub"
[5]: https://forum.manjaro.org/t/zsh-plugins-not-found/170712?utm_source=chatgpt.com "Zsh plugins not found - Software & Applications - Manjaro Linux Forum"
[6]: https://linuxhandbook.com/zsh-auto-suggestion/?utm_source=chatgpt.com "Enabling Auto Suggestion in Zsh - Linux Handbook"