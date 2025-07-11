import random

# --- Transitions for natural flow ---
TRANSITIONS = [
    "Additionally,",
    "Furthermore,",
    "To build further trust,",
    "To deepen engagement,",
    "You might also consider:",
    "Another strategy could be:",
    "In the same vein,",
    "For continued loyalty,",
    "As a next step,",
]

def join_strategies(strategies):
    """Join multiple strategies with natural transitions."""
    if not strategies:
        return "No specific strategy available for selected features."
    return " ".join([strategies[0]] + [f"{random.choice(TRANSITIONS)} {s}" for s in strategies[1:]])

# --- Strategies for each feature per cluster ---
CLUSTER_STRATEGIES = {
    "low": {
        "tenure": [
            "Customer has a short tenure; offer a warm welcome package to increase early engagement.",
            "Provide personalized onboarding sessions to build confidence.",
            "Assign a dedicated account manager for hands-on support.",
            "Highlight easy upgrade options to encourage deeper commitment.",
            "Offer introductory discounts for service add-ons.",
            "Use positive testimonials to boost trust in the early phase.",
            "Send regular check-ins to answer any questions or concerns.",
        ],
        "monthlycharges": [
            "Low monthly charges indicate price sensitivity; suggest affordable add-ons to increase value.",
            "Introduce flexible payment options to reduce financial barriers.",
            "Offer trial periods for premium services at minimal extra cost.",
            "Bundle low-cost entertainment or security options to boost satisfaction.",
            "Highlight any cost-saving plans to reassure budget-conscious users.",
            "Promote loyalty rewards to encourage continued service.",
            "Provide clear cost breakdowns to improve transparency and trust.",
        ],
        "totalcharges": [
            "Total charges are low; encourage gradual upgrades with minimal cost impact.",
            "Introduce bundled service discounts to increase overall value.",
            "Educate customers about premium options to highlight benefits.",
            "Offer flexible upgrade paths that don't disrupt current plans.",
            "Reward usage milestones with exclusive perks.",
            "Communicate how incremental spend leads to better service.",
            "Share success stories of customers who upgraded and saved.",
        ],
        "contract": [
            "Leverage contract stability to introduce loyalty rewards early on.",
            "Encourage early contract renewals with exclusive discounts.",
            "Offer contract flexibility to alleviate commitment fears.",
            "Promote no-penalty trial periods for add-on services.",
            "Highlight benefits of longer contract terms with bonus perks.",
            "Provide contract summaries for clarity and confidence.",
            "Send timely reminders about renewal options and incentives.",
        ],
        "internetservice": [
            "Customer uses internet services; offer slight speed upgrades to increase satisfaction.",
            "Promote value-added services like free modem upgrades.",
            "Offer bundled packages combining internet and entertainment services.",
            "Provide educational content on maximizing internet usage.",
            "Encourage feedback to tailor internet packages individually.",
            "Highlight customer support availability for internet issues.",
            "Offer introductory pricing on premium internet tiers.",
        ],
        "onlinesecurity": [
            "Basic security service users should be encouraged to upgrade to premium protection.",
            "Offer free trials of advanced cybersecurity features.",
            "Educate on risks and benefits of comprehensive security.",
            "Bundle security upgrades with internet service offers.",
            "Provide personalized security assessments for peace of mind.",
            "Offer discounts on multi-device protection plans.",
            "Highlight success stories of customers avoiding threats with upgrades.",
        ],
        "onlinebackup": [
            "Encourage expanding backup storage for critical data safety.",
            "Offer flexible plans based on device number and usage.",
            "Promote peace of mind benefits of cloud backup.",
            "Provide easy-to-understand tutorials on backup restoration.",
            "Bundle backup with security and device protection offers.",
            "Offer time-limited discounts for storage upgrades.",
            "Highlight customer testimonials about backup reliability.",
        ],
        "deviceprotection": [
            "Promote device protection upgrades with bundled insurance offers.",
            "Highlight the convenience and cost-savings of comprehensive device care.",
            "Offer replacement programs to reduce downtime fears.",
            "Provide flexible plans catering to device age and type.",
            "Bundle device protection with tech support for seamless service.",
            "Offer free device health check-ups during subscription renewals.",
            "Send reminders on protection plan expirations with upgrade incentives.",
        ],
        "techsupport": [
            "Promote priority tech support plans to enhance user experience.",
            "Offer fast-track issue resolution as a premium feature.",
            "Bundle tech support with other service upgrades for better value.",
            "Educate customers about troubleshooting resources and support channels.",
            "Provide proactive alerts on common issues and fixes.",
            "Offer remote assistance to resolve problems quickly.",
            "Encourage feedback on support quality to improve services.",
        ],
        "streamingtv": [
            "Streaming TV usage indicates entertainment interest; suggest discounted combo packs.",
            "Offer free trials of premium channels to increase engagement.",
            "Bundle streaming with internet upgrades for better quality.",
            "Provide curated content recommendations based on preferences.",
            "Promote interactive features like watch parties or chat.",
            "Introduce loyalty rewards tied to streaming frequency.",
            "Encourage app downloads for seamless streaming access.",
        ],
        "streamingmovies": [
            "Active movie streaming users could benefit from family-friendly bundles.",
            "Offer themed movie nights or exclusive premieres.",
            "Bundle movie streaming with other entertainment packages.",
            "Promote personalized watchlists to boost engagement.",
            "Provide rewards for frequent viewers and referrals.",
            "Offer flexible subscription tiers for diverse preferences.",
            "Encourage app engagement with push notifications and updates.",
        ],
        "paymentmethod": [
            "Encourage auto-pay enrollment with exclusive discounts.",
            "Offer cashback rewards for consistent payment method use.",
            "Highlight the security and convenience of preferred payment options.",
            "Promote digital wallets or app-based payment methods.",
            "Provide easy payment plan switches to suit customer preferences.",
            "Offer limited-time incentives for trying new payment methods.",
            "Educate about benefits of timely payments and incentives.",
        ],
        "paperlessbilling": [
            "Promote eco-friendly benefits of paperless billing with exclusive offers.",
            "Highlight time-saving mobile features and instant bill access.",
            "Offer discounts or rewards for digital bill adoption.",
            "Educate customers about billing security and privacy.",
            "Bundle paperless billing with app notifications and reminders.",
            "Provide easy sign-up processes for hassle-free adoption.",
            "Send personalized messages about environmental impact benefits.",
        ],
        "partner": [
            "Encourage shared plans and family discounts for partner accounts.",
            "Offer referral bonuses for inviting partners to services.",
            "Promote benefits of joint billing and account management.",
            "Provide couple-friendly content bundles and perks.",
            "Educate about service sharing options and privacy controls.",
            "Offer loyalty bonuses for partner plan renewals.",
            "Highlight success stories of partner-based savings.",
        ],
        "dependents": [
            "Offer multi-device packages suitable for family needs.",
            "Promote parental controls and kid-safe streaming bundles.",
            "Educate parents about educational content offerings.",
            "Provide flexible pricing plans based on dependent numbers.",
            "Offer family loyalty rewards and usage-based perks.",
            "Bundle services appealing to younger demographics.",
            "Send targeted promotions during school holidays or events.",
        ],
        "seniorcitizen": [
            "Offer special discounts tailored for senior citizens.",
            "Promote easy-to-use devices and services designed for seniors.",
            "Provide personalized tech support and tutorials.",
            "Bundle health-related content or emergency service features.",
            "Encourage family plans with senior-focused benefits.",
            "Offer loyalty rewards for long-term senior customers.",
            "Send gentle reminders about usage and support options.",
        ],
    },

    "medium": {
        "tenure": [
            "Customers with moderate tenure value loyalty rewards; offer exclusive perks.",
            "Provide contract renewal benefits to encourage continued service.",
            "Offer service upgrade options tailored to tenure length.",
            "Send personalized messages recognizing customer loyalty.",
            "Promote premium support options to improve satisfaction.",
            "Encourage participation in referral programs.",
            "Provide early access to new features and beta programs.",
        ],
        "monthlycharges": [
            "Medium monthly charges indicate balanced spending; offer value bundles.",
            "Suggest flexible payment plans and service boosts.",
            "Bundle entertainment and security services for better value.",
            "Offer loyalty cashback or discounts for consistent usage.",
            "Provide educational content on cost-effective upgrades.",
            "Promote seasonal offers tied to usage patterns.",
            "Send personalized recommendations for service enhancements.",
        ],
        "totalcharges": [
            "Encourage increased spending through premium perks and upgrades.",
            "Offer invitations to exclusive events and beta features.",
            "Promote bundled plans that increase overall value.",
            "Provide detailed spending summaries highlighting benefits.",
            "Offer milestone rewards for high engagement.",
            "Send proactive reminders about upgrade opportunities.",
            "Educate on advantages of premium services and add-ons.",
        ],
        "contract": [
            "Suggest flexible contract renewals with rolling plan options.",
            "Offer try-before-you-upgrade trial periods.",
            "Highlight benefits of contract extension with bonuses.",
            "Provide clear contract summaries and renewal reminders.",
            "Encourage early renewal with exclusive discounts.",
            "Promote no-penalty upgrades and downgrades.",
            "Send personalized messages explaining contract options.",
        ],
        "internetservice": [
            "Offer internet speed upgrades and modem replacement programs.",
            "Bundle internet with entertainment and security packages.",
            "Provide educational resources on maximizing internet benefits.",
            "Promote customer feedback channels for service improvement.",
            "Offer limited-time promotions on premium internet tiers.",
            "Provide installation and troubleshooting support offers.",
            "Highlight customer success stories using upgraded services.",
        ],
        "onlinesecurity": [
            "Encourage upgrades to full protection plans with discounts.",
            "Promote cybersecurity awareness campaigns.",
            "Offer personalized risk assessments and customized packages.",
            "Bundle security services with other premium plans.",
            "Provide trial periods for advanced security features.",
            "Highlight customer testimonials about improved safety.",
            "Offer proactive alerts and support for security issues.",
        ],
        "onlinebackup": [
            "Upsell expanded backup storage and multi-device options.",
            "Promote ease of data recovery and cloud integration.",
            "Provide tutorial content to boost usage confidence.",
            "Bundle backup with security and device protection plans.",
            "Offer seasonal discounts for storage upgrades.",
            "Highlight reliability benefits through customer stories.",
            "Send proactive reminders to review backup plans.",
        ],
        "deviceprotection": [
            "Offer tiered device protection plans with flexible coverage.",
            "Promote bundling device care with tech support.",
            "Provide replacement guarantees and express service offers.",
            "Highlight cost savings of comprehensive protection plans.",
            "Send reminders about protection plan renewals.",
            "Offer device health check-ups and diagnostics.",
            "Provide easy upgrade paths for device protection plans.",
        ],
        "techsupport": [
            "Promote premium support with guaranteed response times.",
            "Offer remote diagnostics and quick resolution services.",
            "Bundle tech support with other premium services.",
            "Educate on self-service resources and FAQs.",
            "Provide feedback opportunities for support improvement.",
            "Send reminders about support plan expirations.",
            "Offer personalized tech support plans for complex needs.",
        ],
        "streamingtv": [
            "Offer discounted premium channel bundles.",
            "Promote interactive and personalized content options.",
            "Bundle streaming with internet upgrades for better quality.",
            "Provide curated watchlists and content alerts.",
            "Offer loyalty rewards based on viewing habits.",
            "Encourage app usage for enhanced streaming experience.",
            "Send promotional offers for new releases and events.",
        ],
        "streamingmovies": [
            "Suggest flexible movie streaming subscription tiers.",
            "Offer family and group discounts for movie packages.",
            "Promote exclusive premieres and early access.",
            "Provide personalized movie recommendations.",
            "Reward frequent viewers with loyalty bonuses.",
            "Encourage app engagement with notifications.",
            "Offer seasonal promotions on movie packages.",
        ],
        "paymentmethod": [
            "Promote auto-pay with reward points and discounts.",
            "Highlight security features of preferred payment methods.",
            "Offer cashback or bonuses for using digital wallets.",
            "Provide easy options to switch payment methods.",
            "Send reminders for upcoming payment deadlines.",
            "Educate on benefits of consistent, timely payments.",
            "Offer trial promotions for new payment options.",
        ],
        "paperlessbilling": [
            "Encourage adoption of paperless billing with incentives.",
            "Highlight convenience of mobile bill access.",
            "Promote eco-friendly messaging tied to rewards.",
            "Offer tutorials on managing digital bills.",
            "Bundle paperless billing with app notifications.",
            "Send reminders for bill payment through digital channels.",
            "Provide exclusive offers for paperless customers.",
        ],
        "partner": [
            "Promote shared plans with partner-specific discounts.",
            "Offer referral bonuses for adding partners.",
            "Provide joint billing and account management tools.",
            "Highlight couple-friendly content bundles.",
            "Offer loyalty rewards for partner plan renewals.",
            "Educate on privacy and sharing controls.",
            "Send personalized partner engagement offers.",
        ],
        "dependents": [
            "Offer family packages with flexible device limits.",
            "Promote parental controls and child-safe content.",
            "Provide educational service bundles for dependents.",
            "Send targeted promotions during family events.",
            "Offer loyalty rewards for multi-dependent accounts.",
            "Bundle streaming and internet for families.",
            "Encourage feedback to improve family services.",
        ],
        "seniorcitizen": [
            "Offer easy-to-use devices and simplified plans.",
            "Promote senior discounts and loyalty rewards.",
            "Provide personalized tech support and tutorials.",
            "Bundle health and emergency services.",
            "Encourage family plan participation with senior benefits.",
            "Send gentle reminders for support and upgrades.",
            "Highlight success stories from senior users.",
        ],
    },

    "medium-high": {
        "tenure": [
            "Customers with medium-high tenure show loyalty; offer premium loyalty tiers.",
            "Provide exclusive access to new product launches.",
            "Offer personalized upgrade paths reflecting long-term commitment.",
            "Send appreciation gifts or service credits on tenure milestones.",
            "Encourage participation in customer advisory panels.",
            "Offer concierge-level customer service.",
            "Highlight success stories related to long tenure benefits.",
        ],
        "monthlycharges": [
            "Higher monthly charges suggest premium users; offer elite bundles.",
            "Suggest personalized premium upgrades with exclusive content.",
            "Offer dedicated account managers to enhance service.",
            "Promote VIP events and early access to offers.",
            "Provide customized billing options and detailed analytics.",
            "Send periodic reviews of service usage and benefits.",
            "Encourage referrals with high-value rewards.",
        ],
        "totalcharges": [
            "High total charges reflect engagement; reward with premium perks.",
            "Offer invitations to exclusive customer forums and events.",
            "Provide tailored upgrade recommendations based on usage.",
            "Send detailed spending insights with personalized advice.",
            "Offer milestone-based rewards and recognitions.",
            "Promote annual loyalty bonuses and credits.",
            "Highlight testimonials from high-value customers.",
        ],
        "contract": [
            "Offer contract renewal incentives with bonus services.",
            "Provide flexible contract options with upgrade privileges.",
            "Promote loyalty bonuses tied to contract extensions.",
            "Send personalized contract summaries and benefits.",
            "Encourage participation in beta testing new services.",
            "Offer priority service for contract holders.",
            "Provide early renewal discounts and upgrades.",
        ],
        "internetservice": [
            "Offer ultra-fast internet packages with premium support.",
            "Bundle internet with exclusive entertainment and security plans.",
            "Provide proactive service monitoring and issue resolution.",
            "Promote educational webinars on maximizing internet value.",
            "Send personalized upgrade offers and speed tests.",
            "Offer dedicated technical support for premium plans.",
            "Highlight case studies of improved customer experiences.",
        ],
        "onlinesecurity": [
            "Promote enterprise-grade security solutions with VIP support.",
            "Offer customized cybersecurity packages tailored to needs.",
            "Provide proactive threat monitoring and response.",
            "Bundle advanced security with other premium services.",
            "Offer exclusive webinars and security briefings.",
            "Send personalized security health reports.",
            "Encourage feedback on security service enhancements.",
        ],
        "onlinebackup": [
            "Offer unlimited backup storage with multi-device support.",
            "Provide personalized backup recovery planning.",
            "Bundle backup with security and device protection tiers.",
            "Offer priority restoration services.",
            "Send alerts on backup status and recommendations.",
            "Provide tutorials on advanced backup features.",
            "Offer seasonal promotions for backup plan upgrades.",
        ],
        "deviceprotection": [
            "Offer premium device protection with express replacements.",
            "Provide on-site device repair and diagnostics.",
            "Bundle protection with tech support for seamless care.",
            "Send proactive device health reports.",
            "Offer loyalty credits for device protection renewals.",
            "Provide easy upgrade paths to top-tier plans.",
            "Highlight testimonials of premium device care benefits.",
        ],
        "techsupport": [
            "Offer 24/7 dedicated tech support with VIP channels.",
            "Provide on-demand remote assistance and diagnostics.",
            "Bundle tech support with premium product packages.",
            "Send proactive support check-ins and issue prevention tips.",
            "Offer personalized tech tutorials and FAQs.",
            "Encourage feedback on support satisfaction.",
            "Provide priority scheduling for complex issues.",
        ],
        "streamingtv": [
            "Offer curated premium channel bundles with exclusive content.",
            "Promote interactive streaming features and social viewing.",
            "Bundle streaming with ultra-fast internet for best experience.",
            "Provide personalized content curation services.",
            "Offer loyalty rewards based on usage milestones.",
            "Send invitations to exclusive streaming events.",
            "Encourage use of apps and smart devices for streaming.",
        ],
        "streamingmovies": [
            "Offer premium movie streaming subscriptions with early releases.",
            "Provide family and group plans with added benefits.",
            "Promote exclusive premieres and VIP viewing parties.",
            "Send personalized movie recommendations and alerts.",
            "Offer loyalty bonuses for frequent viewers.",
            "Encourage app engagement with notifications and updates.",
            "Provide seasonal discounts on movie bundles.",
        ],
        "paymentmethod": [
            "Promote premium payment options with exclusive rewards.",
            "Offer concierge billing support and personalized payment plans.",
            "Highlight security and convenience of advanced payment methods.",
            "Provide cashback and loyalty points for consistent use.",
            "Send reminders and updates on payment benefits.",
            "Offer trials for new payment technologies.",
            "Educate on benefits of timely payments and credit building.",
        ],
        "paperlessbilling": [
            "Promote premium paperless billing services with detailed analytics.",
            "Offer rewards and incentives for digital billing adoption.",
            "Provide advanced mobile billing and payment notifications.",
            "Bundle paperless billing with premium support services.",
            "Send personalized usage and billing reports.",
            "Encourage feedback on digital billing improvements.",
            "Highlight environmental and convenience benefits.",
        ],
        "partner": [
            "Offer premium shared plans with exclusive partner discounts.",
            "Provide joint account management with enhanced features.",
            "Promote couple and family loyalty programs.",
            "Send personalized offers for partner engagement.",
            "Encourage referrals with high-value rewards.",
            "Offer privacy and sharing controls tailored to partners.",
            "Highlight success stories of partner benefits.",
        ],
        "dependents": [
            "Offer premium family packages with multi-device and content options.",
            "Promote educational and child-safe content bundles.",
            "Provide personalized support for family account management.",
            "Send targeted offers for family events and holidays.",
            "Offer loyalty bonuses for multi-dependent plans.",
            "Encourage feedback to optimize family services.",
            "Highlight case studies of successful family plans.",
        ],
        "seniorcitizen": [
            "Offer advanced tech support and senior-friendly devices.",
            "Provide personalized health and safety content bundles.",
            "Promote senior loyalty rewards and discounts.",
            "Send gentle reminders and personalized assistance offers.",
            "Encourage family involvement in service plans.",
            "Provide easy-to-understand tutorials and resources.",
            "Highlight testimonials from satisfied senior customers.",
        ],
    },

    "high": {
        "tenure": [
            "High-tenure customers are brand champions; offer exclusive VIP programs.",
            "Send personalized thank-you gifts and service credits.",
            "Invite to beta-test new products and services.",
            "Provide direct access to senior support teams.",
            "Offer premium renewal bonuses and lifetime discounts.",
            "Highlight their role in shaping company offerings.",
            "Encourage participation in high-level customer advisory boards.",
        ],
        "monthlycharges": [
            "Top-tier monthly spenders deserve elite offers and concierge services.",
            "Provide customized premium content and service bundles.",
            "Offer early access to exclusive products and events.",
            "Send personalized billing reviews and optimization suggestions.",
            "Provide dedicated account managers and rapid support.",
            "Offer referral incentives with significant rewards.",
            "Promote exclusive membership perks and loyalty tiers.",
        ],
        "totalcharges": [
            "High overall spend reflects strong loyalty; reward with elite privileges.",
            "Invite to exclusive customer events and forums.",
            "Offer bespoke upgrade paths with personal consultations.",
            "Send detailed insights into service usage and benefits.",
            "Provide milestone awards and public recognition options.",
            "Offer lifetime loyalty discounts and bonuses.",
            "Highlight case studies of VIP customer benefits.",
        ],
        "contract": [
            "Offer lifetime contract options with premium benefits.",
            "Provide concierge renewal services and upgrade consultations.",
            "Send personalized contract reviews and recommendations.",
            "Offer no-penalty flexible contract adjustments.",
            "Promote exclusive contract extension bonuses.",
            "Provide direct lines to contract managers.",
            "Encourage participation in contract advisory groups.",
        ],
        "internetservice": [
            "Offer ultra-premium internet packages with SLA guarantees.",
            "Provide dedicated technical support and proactive monitoring.",
            "Bundle with exclusive entertainment and security services.",
            "Send personalized speed optimization reports.",
            "Offer VIP installation and upgrade services.",
            "Provide early access to internet innovations.",
            "Highlight success stories of premium internet users.",
        ],
        "onlinesecurity": [
            "Offer top-tier cybersecurity with dedicated threat response teams.",
            "Provide customized security audits and protection plans.",
            "Bundle with premium device protection and tech support.",
            "Send real-time security alerts and personal consultations.",
            "Offer exclusive webinars on latest security trends.",
            "Provide personalized risk mitigation strategies.",
            "Encourage feedback for continuous security enhancement.",
        ],
        "onlinebackup": [
            "Offer unlimited premium backup with instant recovery.",
            "Provide concierge backup support and monitoring.",
            "Bundle backup with top-tier security and device care.",
            "Send proactive backup health reports and tips.",
            "Offer early access to new backup features.",
            "Provide personalized data protection consultations.",
            "Highlight testimonials from premium backup customers.",
        ],
        "deviceprotection": [
            "Offer VIP device protection with on-demand replacements.",
            "Provide on-site premium repair and diagnostics.",
            "Bundle with exclusive tech support services.",
            "Send proactive device health monitoring reports.",
            "Offer lifetime device protection plans.",
            "Provide loyalty bonuses for continuous protection renewals.",
            "Highlight case studies of premium device care success.",
        ],
        "techsupport": [
            "Provide 24/7 dedicated VIP tech support teams.",
            "Offer on-demand expert assistance and diagnostics.",
            "Bundle tech support with elite product packages.",
            "Send personalized support follow-ups and satisfaction surveys.",
            "Provide priority scheduling and rapid issue resolution.",
            "Offer exclusive tutorials and resources.",
            "Encourage VIP customer feedback programs.",
        ],
        "streamingtv": [
            "Offer exclusive premium channel bundles with VIP content.",
            "Provide interactive social streaming experiences.",
            "Bundle with ultra-fast internet and premium devices.",
            "Send personalized content curation and early release access.",
            "Offer VIP loyalty rewards and invitations to events.",
            "Encourage premium app usage with exclusive features.",
            "Highlight case studies of premium streaming users.",
        ],
        "streamingmovies": [
            "Offer premium movie streaming with early premieres.",
            "Provide family/group VIP plans with exclusive benefits.",
            "Send invitations to private viewing parties.",
            "Offer personalized movie recommendations and alerts.",
            "Provide loyalty rewards for frequent premium viewers.",
            "Encourage premium app engagement with notifications.",
            "Offer seasonal VIP discounts and bundles.",
        ],
        "paymentmethod": [
            "Provide concierge billing and premium payment plans.",
            "Offer cashback and exclusive rewards for top-tier methods.",
            "Highlight enhanced security for premium payment options.",
            "Send personalized payment optimization advice.",
            "Offer early access to new payment technologies.",
            "Provide dedicated support for payment inquiries.",
            "Encourage loyalty through premium payment incentives.",
        ],
        "paperlessbilling": [
            "Offer advanced analytics and detailed digital billing reports.",
            "Provide exclusive rewards for paperless billing loyalty.",
            "Bundle with premium support and mobile notifications.",
            "Send personalized usage summaries and savings insights.",
            "Promote environmental leadership among VIP customers.",
            "Encourage feedback to enhance digital billing features.",
            "Highlight success stories of paperless billing adopters.",
        ],
        "partner": [
            "Offer premium shared plans with exclusive partner perks.",
            "Provide joint account management with enhanced controls.",
            "Promote couple/family VIP loyalty programs.",
            "Send personalized partner engagement and reward offers.",
            "Encourage high-value referrals with elite bonuses.",
            "Offer privacy and sharing settings tailored for VIPs.",
            "Highlight partner success stories and testimonials.",
        ],
        "dependents": [
            "Offer premium family plans with extensive device and content options.",
            "Provide personalized educational and safety content bundles.",
            "Send targeted VIP offers during family events.",
            "Offer loyalty bonuses for high-value family accounts.",
            "Encourage family feedback to optimize premium services.",
            "Highlight case studies of successful premium family plans.",
            "Provide concierge family support and account management.",
        ],
        "seniorcitizen": [
            "Offer VIP tech support and senior-friendly premium devices.",
            "Provide personalized health and safety service bundles.",
            "Promote senior loyalty rewards with exclusive discounts.",
            "Send gentle reminders and VIP personalized assistance.",
            "Encourage family participation in senior service plans.",
            "Provide easy-to-understand tutorials tailored for seniors.",
            "Highlight testimonials from satisfied VIP senior customers.",
        ],
    },
}

def generate_retention_strategy(cluster, features):
    """
    Generate retention strategies grouped by feature,
    each bullet point is a flowing paragraph (3-4 sentences).
    """
    paragraphs = []
    cluster = cluster.lower()

    for feature in features:
        if feature not in CLUSTER_STRATEGIES.get(cluster, {}):
            paragraphs.append(f"No strategies available for feature '{feature}'.")
            continue

        strategies_list = CLUSTER_STRATEGIES[cluster][feature]

        # We'll create multiple strategy paragraphs, each with 3-4 sentences
        # For that, randomly sample 6-7 sentences and group every 3-4 sentences as one paragraph
        # Or simpler: make 2-3 paragraphs by picking 3-4 sentences each time

        all_sentences = strategies_list[:]  # copy list
        random.shuffle(all_sentences)

        bullet_paragraphs = []
        # Pick 2 or 3 paragraphs depending on available sentences
        num_paragraphs = min(3, max(1, len(all_sentences) // 4))

        for i in range(num_paragraphs):
            # pick 3-4 sentences for one paragraph
            start_idx = i * 4
            end_idx = start_idx + 4
            sentences_chunk = all_sentences[start_idx:end_idx]

            # join them into a flowing paragraph
            paragraph = join_strategies(sentences_chunk)
            bullet_paragraphs.append(paragraph)

        # Format bullet points with full paragraphs
        feature_bullets = "\n".join([f"- {bp}" for bp in bullet_paragraphs])
        paragraphs.append(feature_bullets)

    return "\n\n".join(paragraphs)

if __name__ == "__main__":
    # For quick test/demo:
    features_name = [
        "tenure", "monthlycharges", "totalcharges", "contract", "internetservice",
        "onlinesecurity", "onlinebackup", "deviceprotection", "techsupport",
        "streamingtv", "streamingmovies", "paymentmethod", "paperlessbilling",
        "partner", "dependents", "seniorcitizen"
    ]
    cluster = "low"  # Can be low, medium, medhigh, high

    print(f"Retention strategies for cluster '{cluster}':\n")
    print(generate_feature_strategies(cluster, features_name))
