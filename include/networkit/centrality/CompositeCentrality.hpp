/*
 * CompositeCentrality.hpp
 *
 *  Created on: 15.11.2026
 *      Author: Prawal Pokharel
 */

// networkit-format

#ifndef NETWORKIT_CENTRALITY_COMPOSITE_CENTRALITY_HPP_
#define NETWORKIT_CENTRALITY_COMPOSITE_CENTRALITY_HPP_

#include <vector>

#include <networkit/centrality/Centrality.hpp>

namespace NetworKit {

/**
 * @ingroup centrality
 * Weighted composite of four standard centralities, optionally combined with the entropy of a
 * per-node demand history and a per-node attribute. This is the Centrality-Entropy Index (CEI)
 * of P. Pokharel, "Evaluation of the CEI Framework Enterprise Cloud Governance in Multi-Tier
 * Distributed Environments", Future Technologies Conference (FTC) 2026, to appear:
 *
 * $$ s(v) = \alpha C(v) + \beta H(v) + \gamma R(v) $$
 *
 * - C(v) is a weighted sum of PageRank, betweenness, degree centrality and closeness, divided by
 *   its maximum over all nodes. The default weights are 0.35, 0.30, 0.20 and 0.15.
 * - H(v) is the Shannon entropy of the node's demand history (values in [0, 1], put into
 *   equal-width bins), divided by log2(bins). It is 0 for nodes without a history.
 * - R(v) is a per-node attribute, for example an operator-declared criticality in [0, 1].
 *   It is 0 if no attributes are set.
 *
 * Without histories and attributes, the score is alpha times the composite C.
 *
 * By default, the four components use the normalisations of NetworkX: PageRank sums to 1 (sinks
 * distributed uniformly), betweenness is normalised by the number of node pairs, degree
 * centrality is deg(v) / (n - 1) (in-degree plus out-degree on directed graphs), and closeness
 * is the Wasserman-Faust variant for disconnected graphs, using inbound distances on directed
 * graphs. Their scales differ, so the component weights do not set each component's share.
 * Set @a scaleComponents to divide each component by its maximum before weighting.
 */
class CompositeCentrality final : public Centrality {
public:
    /**
     * How betweenness is computed: exactly (Brandes, O(nm)), by path sampling
     * (ApproxBetweenness), or by adaptive sampling (KadabraBetweenness).
     */
    enum class BetweennessMethod { EXACT, APPROXIMATE, KADABRA };

    /**
     * How closeness is computed: exactly (O(nm)) or by sampling (ApproxCloseness).
     */
    enum class ClosenessMethod { EXACT, APPROXIMATE };

    /**
     * Component index for componentScores().
     */
    enum class Component { PAGERANK, BETWEENNESS, DEGREE, CLOSENESS };

    /**
     * Constructs the CompositeCentrality class for the given graph @a G.
     *
     * @param G The graph. Edge weights are ignored.
     * @param alpha Weight of the composite centrality C.
     * @param beta Weight of the demand entropy H.
     * @param gamma Weight of the node attribute R.
     * @param scaleComponents If true, each of the four centralities is divided by its maximum
     * before the component weights are applied.
     */
    CompositeCentrality(const Graph &G, double alpha = 0.40, double beta = 0.35,
                        double gamma = 0.25, bool scaleComponents = false);

    /**
     * Sets the weights of PageRank, betweenness, degree centrality and closeness in C.
     * A component with weight 0 is not computed. Weights must be non-negative and must not all
     * be 0.
     */
    void setComponentWeights(double pageRank, double betweenness, double degree, double closeness);

    /**
     * Sets how betweenness is computed. @a epsilon and @a delta are the additive error and the
     * failure probability of the approximate methods. Approximate scores are rescaled to the
     * normalisation of the exact method. The approximate methods need an undirected graph;
     * run() throws std::invalid_argument on a directed one.
     */
    void setBetweennessMethod(BetweennessMethod method, double epsilon = 0.01, double delta = 0.1);

    /**
     * Sets how closeness is computed. @a samples is the number of samples of the approximate
     * method. The approximate method is only available on connected undirected graphs.
     */
    void setClosenessMethod(ClosenessMethod method, count samples = 100);

    /**
     * Sets the demand history of every node, indexed by node id (size G.upperNodeIdBound()).
     * Values are clipped to [0, 1] and counted in @a bins equal-width bins.
     */
    void setDemandHistories(std::vector<std::vector<double>> histories, count bins = 10);

    /**
     * Sets the attribute R of every node, indexed by node id (size G.upperNodeIdBound()).
     */
    void setNodeAttributes(std::vector<double> attributes);

    void run() override;

    /**
     * Returns alpha + beta + gamma, the largest score a node can reach.
     */
    double maximum() override;

    /**
     * Returns C, the composite centrality divided by its maximum.
     */
    const std::vector<double> &centralityScores() const;

    /**
     * Returns H, the normalised demand entropy (0 for nodes without a history).
     */
    const std::vector<double> &entropyScores() const;

    /**
     * Returns R, the node attributes (0 if none were set).
     */
    const std::vector<double> &attributeScores() const;

    /**
     * Returns one of the four centralities as it entered C (after scaling, if enabled).
     * Components with weight 0 are all zeros.
     */
    const std::vector<double> &componentScores(Component component) const;

    /**
     * Normalised Shannon entropy of a series of values in [0, 1] over @a bins equal-width bins,
     * with the binning rules of numpy.histogram. Returns 0 for an empty series.
     */
    static double demandEntropy(const std::vector<double> &values, count bins = 10);

private:
    double alpha, beta, gamma;
    bool scaleComponents;
    double wPageRank = 0.35, wBetweenness = 0.30, wDegree = 0.20, wCloseness = 0.15;
    BetweennessMethod betweennessMethod = BetweennessMethod::EXACT;
    double epsilon = 0.01, delta = 0.1;
    ClosenessMethod closenessMethod = ClosenessMethod::EXACT;
    count closenessSamples = 100;
    std::vector<std::vector<double>> histories;
    count bins = 10;
    std::vector<double> attributes;

    std::vector<double> pageRankData, betweennessData, degreeData, closenessData;
    std::vector<double> centralityData, entropyData, attributeData;

    void computePageRank(const Graph &H);
    void computeBetweenness(const Graph &H);
    void computeDegree(const Graph &H);
    void computeCloseness(const Graph &H);
};

} // namespace NetworKit

#endif // NETWORKIT_CENTRALITY_COMPOSITE_CENTRALITY_HPP_
