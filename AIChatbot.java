import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class AIChatbot extends JFrame {
    // ===== Data structures =====
    static class FAQ {
        String question;
        String answer;
        FAQ(String q, String a){ this.question=q; this.answer=a; }
    }

    private final JTextArea chatArea = new JTextArea();
    private final JTextField inputField = new JTextField();
    private final JButton sendBtn = new JButton("Send");
    private final JButton addFaqBtn = new JButton("Add FAQ");
    private final JButton showFaqsBtn = new JButton("Show FAQs");
    private final JLabel statusLabel = new JLabel("Ready");

    private final List<FAQ> faqs = new ArrayList<>();
    private final Set<String> stopwords = new HashSet<>(Arrays.asList(
            "a","an","the","and","or","but","if","then","else","when","at","by","for","with",
            "about","against","between","into","through","during","before","after","above","below",
            "to","from","up","down","in","out","on","off","over","under","again","further","here",
            "there","why","how","all","any","both","each","few","more","most","other","some","such",
            "no","nor","not","only","own","same","so","than","too","very","can","will","just","is",
            "am","are","was","were","be","been","being","do","does","did","doing","of"
    ));

    // Vocabulary & IDF for TF-IDF
    private Map<String, Double> idf = new HashMap<>();
    private List<Map<String, Double>> faqTfIdfVectors = new ArrayList<>();
    private Set<String> vocabulary = new HashSet<>();

    private static final String FAQ_FILE = "faqs.tsv";
    private static final double SIM_THRESHOLD = 0.22; // tune threshold

    public AIChatbot() {
        super("AI Chatbot — FAQ + NLP");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(800, 540);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout(10,10));

        // Chat area
        chatArea.setEditable(false);
        chatArea.setLineWrap(true);
        chatArea.setWrapStyleWord(true);
        JScrollPane sp = new JScrollPane(chatArea);
        add(sp, BorderLayout.CENTER);

        // Input panel
        JPanel inputPanel = new JPanel(new BorderLayout(8,8));
        inputPanel.add(inputField, BorderLayout.CENTER);
        JPanel btns = new JPanel(new FlowLayout(FlowLayout.RIGHT,8,8));
        btns.add(addFaqBtn);
        btns.add(showFaqsBtn);
        btns.add(sendBtn);
        inputPanel.add(btns, BorderLayout.EAST);
        add(inputPanel, BorderLayout.SOUTH);

        // Status
        JPanel top = new JPanel(new BorderLayout());
        top.add(statusLabel, BorderLayout.WEST);
        add(top, BorderLayout.NORTH);

        // Events
        sendBtn.addActionListener(e -> handleUserSend());
        inputField.addActionListener(e -> handleUserSend());
        addFaqBtn.addActionListener(e -> openAddFaqDialog());
        showFaqsBtn.addActionListener(e -> showFaqList());

        // Load + seed FAQs, then build TF-IDF
        loadOrSeedFAQs();
        rebuildTfIdf();

        // Welcome
        botSay("Hi! I’m your AI assistant 🤖.\n" +
               "Ask me anything, or add FAQs via the 'Add FAQ' button.\n" +
               "Type 'help' for tips.");
    }

    // ===== Core interaction =====
    private void handleUserSend() {
        String userText = inputField.getText().trim();
        if (userText.isEmpty()) return;
        userSay(userText);
        inputField.setText("");

        String response = respond(userText);
        botSay(response);
    }

    private void userSay(String text)   { chatArea.append("You: " + text + "\n"); }
    private void botSay(String text)    { chatArea.append("Bot: " + text + "\n\n"); }

    // ===== NLP pipeline =====
    private List<String> tokenize(String text) {
        // Lowercase, remove punctuation, split on whitespace
        String normalized = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ");
        String[] toks = normalized.trim().split("\\s+");
        List<String> list = new ArrayList<>();
        for (String t : toks) {
            if (t.isEmpty()) continue;
            if (!stopwords.contains(t)) list.add(t);
        }
        return list;
    }

    private Map<String, Double> termFrequency(List<String> tokens) {
        Map<String, Double> tf = new HashMap<>();
        for (String tok : tokens) tf.put(tok, tf.getOrDefault(tok, 0.0)+1.0);
        double max = tf.values().stream().mapToDouble(d->d).max().orElse(1.0);
        tf.replaceAll((k,v) -> v / max); // normalized term freq
        return tf;
    }

    private void rebuildTfIdf() {
        // Build vocabulary from FAQ questions
        vocabulary.clear();
        List<List<String>> faqTokens = new ArrayList<>();
        for (FAQ f : faqs) {
            List<String> toks = tokenize(f.question);
            faqTokens.add(toks);
            vocabulary.addAll(toks);
        }

        // Compute DF
        Map<String, Integer> df = new HashMap<>();
        for (List<String> doc : faqTokens) {
            Set<String> uniq = new HashSet<>(doc);
            for (String w : uniq) df.put(w, df.getOrDefault(w, 0)+1);
        }

        int N = Math.max(1, faqTokens.size());
        idf.clear();
        for (String w : vocabulary) {
            int dfi = df.getOrDefault(w, 0);
            // Smooth IDF
            double val = Math.log((N + 1.0) / (dfi + 1.0)) + 1.0;
            idf.put(w, val);
        }

        // Build TF-IDF vectors for FAQs
        faqTfIdfVectors.clear();
        for (List<String> doc : faqTokens) {
            Map<String, Double> tf = termFrequency(doc);
            Map<String, Double> vec = new HashMap<>();
            for (Map.Entry<String, Double> e : tf.entrySet()) {
                double wIdf = idf.getOrDefault(e.getKey(), 1.0);
                vec.put(e.getKey(), e.getValue() * wIdf);
            }
            faqTfIdfVectors.add(vec);
        }
        statusLabel.setText("FAQs: " + faqs.size());
    }

    private double cosineSim(Map<String, Double> v1, Map<String, Double> v2) {
        // dot
        double dot = 0.0;
        // iterate over smaller map
        Map<String, Double> a = v1.size() < v2.size() ? v1 : v2;
        Map<String, Double> b = a == v1 ? v2 : v1;
        for (Map.Entry<String, Double> e : a.entrySet()) {
            Double bv = b.get(e.getKey());
            if (bv != null) dot += e.getValue() * bv;
        }
        // norms
        double n1 = 0.0, n2 = 0.0;
        for (double d : v1.values()) n1 += d*d;
        for (double d : v2.values()) n2 += d*d;
        if (n1 == 0 || n2 == 0) return 0.0;
        return dot / (Math.sqrt(n1) * Math.sqrt(n2));
    }

    // ===== Hybrid response engine =====
    private String respond(String userText) {
        String quick = ruleBased(userText);
        if (quick != null) return quick;

        // FAQ Retrieval via TF-IDF
        String bestAnswer = null;
        double bestSim = -1.0;

        Map<String, Double> qVec = toTfIdfVector(userText);
        for (int i = 0; i < faqs.size(); i++) {
            double sim = cosineSim(qVec, faqTfIdfVectors.get(i));
            if (sim > bestSim) {
                bestSim = sim;
                bestAnswer = faqs.get(i).answer;
            }
        }

        if (bestSim >= SIM_THRESHOLD) {
            return bestAnswer + note(String.format(" (matched via FAQ, sim=%.2f)", bestSim));
        }

        // Sentiment-aware nudge (very simple)
        String sentiment = sentiment(userText);
        if ("neg".equals(sentiment)) {
            return "I'm sorry this is frustrating. Could you rephrase your question or give me a bit more detail? I can also learn it—click 'Add FAQ' to teach me.";
        }

        return "I’m not sure yet 🤔. Try rephrasing, or use **Add FAQ** to teach me the answer for next time!";
    }

    private Map<String, Double> toTfIdfVector(String text) {
        List<String> toks = tokenize(text);
        Map<String, Double> tf = termFrequency(toks);
        Map<String, Double> vec = new HashMap<>();
        for (Map.Entry<String, Double> e : tf.entrySet()) {
            double wIdf = idf.getOrDefault(e.getKey(), 1.0);
            vec.put(e.getKey(), e.getValue() * wIdf);
        }
        return vec;
    }

    // ===== Rule-based layer =====
    private String ruleBased(String text) {
        String t = text.trim().toLowerCase();

        // Help
        if (t.equals("help") || t.equals("menu")) {
            return "You can ask about timings, features, simple how-tos, or FAQs.\n" +
                   "Commands: 'clear' to clear chat, 'show faqs' to list.\n" +
                   "Use 'Add FAQ' to teach me new Q&A.";
        }

        // Clear
        if (t.equals("clear")) {
            chatArea.setText("");
            return "Chat cleared ✅";
        }

        // Greetings
        if (t.matches("^(hi|hello|hey|hola|namaste|good (morning|afternoon|evening))\\b.*")) {
            return "Hello! How can I help you today?";
        }

        // Farewell
        if (t.matches("^(bye|goodbye|see you|thanks|thank you)\\b.*")) {
            return "You're welcome! Have a great day 👋";
        }

        // Time/date
        if (t.contains("time")) {
            return "Current time: " + new Date().toString();
        }

        // Name
        if (t.contains("your name")) {
            return "I’m a lightweight Java AI Chatbot. You can train me using the Add FAQ button!";
        }

        // Simple “what can you do”
        if (t.contains("what can you do") || t.contains("features")) {
            return "I support real-time chat, FAQ retrieval (TF‑IDF), rule-based replies, and on-the-fly training with persistent storage.";
        }

        // Show FAQs (via message)
        if (t.equals("show faqs")) {
            return faqs.isEmpty() ? "No FAQs yet. Add some with the button!"
                    : "I know " + faqs.size() + " FAQs. Click 'Show FAQs' to view them in a window.";
        }

        return null; // fall through to retrieval
    }

    // Very light sentiment cue
    private String sentiment(String text) {
        String tl = text.toLowerCase();
        String[] neg = {"bad","hate","worst","angry","upset","error","issue","problem","slow","fail","not working"};
        String[] pos = {"good","great","love","awesome","excellent","nice","fast","thanks","thank you"};
        int score = 0;
        for (String w : pos) if (tl.contains(w)) score++;
        for (String w : neg) if (tl.contains(w)) score--;
        return score < 0 ? "neg" : (score > 0 ? "pos" : "neu");
    }

    private String note(String s){ return ""; /* hide debug note in final replies; set to s to show */ }

    // ===== FAQ storage (train at runtime) =====
    private void loadOrSeedFAQs() {
        // Try load file
        if (Files.exists(Path.of(FAQ_FILE))) {
            try (BufferedReader br = Files.newBufferedReader(Path.of(FAQ_FILE))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split("\t", 2);
                    if (parts.length == 2 && !parts[0].isBlank() && !parts[1].isBlank()) {
                        faqs.add(new FAQ(parts[0].trim(), parts[1].trim()));
                    }
                }
            } catch (IOException e) {
                System.err.println("Failed to read faqs.tsv: " + e.getMessage());
            }
        }

        // Seed a few defaults if empty
        if (faqs.isEmpty()) {
            faqs.add(new FAQ("what is your purpose",
                    "I answer common questions and learn FAQs you add in the app."));
            faqs.add(new FAQ("how do i add a new faq",
                    "Click the 'Add FAQ' button, enter a question and answer, and save."));
            faqs.add(new FAQ("how to clear the chat",
                    "Type 'clear' and press Enter, or click the 'Send' after typing clear."));
            faqs.add(new FAQ("what nlp do you use",
                    "I use tokenization, stopword removal, and TF-IDF similarity to find best matching FAQ."));
            persistFAQs(); // write seeds
        }
    }

    private void persistFAQs() {
        try (BufferedWriter bw = Files.newBufferedWriter(Path.of(FAQ_FILE))) {
            for (FAQ f : faqs) {
                bw.write(f.question.replace("\t"," ").trim() + "\t" + f.answer.replace("\t"," ").trim());
                bw.newLine();
            }
        } catch (IOException e) {
            System.err.println("Failed to write faqs.tsv: " + e.getMessage());
        }
    }

    private void openAddFaqDialog() {
        JTextField qField = new JTextField();
        JTextArea aArea = new JTextArea(6, 30);
        aArea.setLineWrap(true);
        aArea.setWrapStyleWord(true);

        JPanel panel = new JPanel(new BorderLayout(8,8));
        JPanel labels = new JPanel(new GridLayout(2,1,6,6));
        labels.add(new JLabel("Question:"));
        labels.add(new JLabel("Answer:"));
        JPanel inputs = new JPanel(new GridLayout(2,1,6,6));
        inputs.add(qField);
        inputs.add(new JScrollPane(aArea));
        panel.add(labels, BorderLayout.WEST);
        panel.add(inputs, BorderLayout.CENTER);

        int res = JOptionPane.showConfirmDialog(this, panel, "Add FAQ", JOptionPane.OK_CANCEL_OPTION, JOptionPane.PLAIN_MESSAGE);
        if (res == JOptionPane.OK_OPTION) {
            String q = qField.getText().trim();
            String a = aArea.getText().trim();
            if (q.isEmpty() || a.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Both question and answer are required.");
                return;
            }
            faqs.add(new FAQ(q, a));
            persistFAQs();
            rebuildTfIdf();
            botSay("Learned new FAQ ✅\nQ: " + q + "\nA: " + a);
        }
    }

    private void showFaqList() {
        if (faqs.isEmpty()) {
            JOptionPane.showMessageDialog(this, "No FAQs available.");
            return;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < faqs.size(); i++) {
            sb.append(i+1).append(". Q: ").append(faqs.get(i).question).append("\n   A: ").append(faqs.get(i).answer).append("\n\n");
        }
        JTextArea area = new JTextArea(sb.toString(), 20, 50);
        area.setEditable(false);
        area.setLineWrap(true);
        area.setWrapStyleWord(true);
        JScrollPane jsp = new JScrollPane(area);
        JOptionPane.showMessageDialog(this, jsp, "FAQs (" + faqs.size() + ")", JOptionPane.PLAIN_MESSAGE);
    }

    // ===== Main =====
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new AIChatbot().setVisible(true));
    }
}
