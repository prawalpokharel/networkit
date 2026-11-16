/*
 * CompositeCentrality.cpp
 *
 *  Created on: 15.11.2026
 *      Author: Prawal Pokharel
 */

// networkit-format

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <networkit/centrality/ApproxBetweenness.hpp>
#include <networkit/centrality/ApproxCloseness.hpp>
#include <networkit/centrality/Betweenness.hpp>
#include <networkit/centrality/Closeness.hpp>
#include <networkit/centrality/CompositeCentrality.hpp>
#include <networkit/centrality/KadabraBetweenness.hpp>
#include <networkit/centrality/PageRank.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/components/StronglyConnectedComponents.hpp>
#include <networkit/graph/GraphTools.hpp>

namespace NetworKit {

namespace {

void requireNonNegative(double value, const char *name) {
    if (!(value >= 0.))
        throw std::invalid_argument(std::string(name) + " must be non-negative");
}

// Divides every entry by the largest entry over existing nodes, if that is positive.
void divideByMax(const Graph &G, std::vector<double> &values) {
    double best = 0.;
    G.forNodes([&](node u) { best = std::max(best, values[u]); });
    if (best > 0.)
        G.parallelForNodes([&](node u) { values[u] /= best; });
}

} // namespace

CompositeCentrality::CompositeCentrality(const Graph &G, double alpha, double beta, double gamma,
                                         bool scaleComponents)
    : Centrality(G, true), alpha(alpha), beta(beta), gamma(gamma),
      scaleComponents(scaleComponents) {
    requireNonNegative(alpha, "alpha");
    requireNonNegative(beta, "beta");
    requireNonNegative(gamma, "gamma");
}

void CompositeCentrality::setComponentWeights(double pageRank, double betweenness, double degree,
                                              double closeness) {
    requireNonNegative(pageRank, "PageRank weight");
    requireNonNegative(betweenness, "betweenness weight");
    requireNonNegative(degree, "degree weight");
    requireNonNegative(closeness, "closeness weight");
    if (pageRank + betweenness + degree + closeness <= 0.)
        throw std::invalid_argument("at least one component weight must be positive");
    wPageRank = pageRank;
    wBetweenness = betweenness;
    wDegree = degree;
    wCloseness = closeness;
}

void CompositeCentrality::setBetweennessMethod(BetweennessMethod method, double epsilon,
                                               double delta) {
    if (!(epsilon > 0. && epsilon < 1.) || !(delta > 0. && delta < 1.))
        throw std::invalid_argument("epsilon and delta must be in (0, 1)");
    betweennessMethod = method;
    this->epsilon = epsilon;
    this->delta = delta;
}

void CompositeCentrality::setClosenessMethod(ClosenessMethod method, count samples) {
    if (samples == 0)
        throw std::invalid_argument("the number of closeness samples must be positive");
    closenessMethod = method;
    closenessSamples = samples;
}

void CompositeCentrality::setDemandHistories(std::vector<std::vector<double>> histories,
                                             count bins) {
    if (!histories.empty() && histories.size() != G.upperNodeIdBound())
        throw std::invalid_argument("histories must have one entry per node id");
    if (bins == 0)
        throw std::invalid_argument("the number of bins must be positive");
    this->histories = std::move(histories);
    this->bins = bins;
}

void CompositeCentrality::setNodeAttributes(std::vector<double> attributes) {
    if (!attributes.empty() && attributes.size() != G.upperNodeIdBound())
        throw std::invalid_argument("attributes must have one entry per node id");
    this->attributes = std::move(attributes);
}

double CompositeCentrality::demandEntropy(const std::vector<double> &values, count bins) {
    if (values.empty() || bins == 0)
        return 0.;
    // Bin edges and index corrections follow numpy.histogram(values, bins, range=(0, 1)),
    // so that results match NumPy-based implementations exactly.
    std::vector<double> edges(bins + 1);
    const double step = 1. / static_cast<double>(bins);
    for (count i = 0; i < bins; ++i)
        edges[i] = static_cast<double>(i) * step;
    edges[bins] = 1.;

    std::vector<count> hist(bins, 0);
    for (double x : values) {
        x = std::clamp(x, 0., 1.);
        auto idx = static_cast<count>(x * static_cast<double>(bins));
        if (idx == bins)
            --idx;
        if (x < edges[idx])
            --idx;
        else if (idx != bins - 1 && x >= edges[idx + 1])
            ++idx;
        ++hist[idx];
    }

    const auto total = static_cast<double>(values.size());
    double entropy = 0.;
    for (count c : hist) {
        if (c == 0)
            continue;
        const double p = static_cast<double>(c) / total;
        entropy -= p * std::log2(p);
    }
    if (bins == 1)
        return 0.;
    return std::min(1., entropy / std::log2(static_cast<double>(bins)));
}

void CompositeCentrality::computePageRank(const Graph &H) {
    pageRankData.assign(H.upperNodeIdBound(), 0.);
    if (wPageRank == 0. || H.numberOfNodes() == 0)
        return;
    // NetworkX treats isolated nodes of undirected graphs as sinks. NetworKit distributes sinks
    // only on directed graphs, so such graphs are run as their directed equivalent.
    bool hasIsolated = false;
    if (!H.isDirected())
        H.forNodes([&](node u) { hasIsolated = hasIsolated || H.degree(u) == 0; });
    std::unique_ptr<Graph> directedCopy;
    if (hasIsolated)
        directedCopy = std::make_unique<Graph>(H, false, true);
    const Graph &P = hasIsolated ? *directedCopy : H;
    PageRank pr(P, 0.85, 1e-9, false, PageRank::SinkHandling::DISTRIBUTE_SINKS);
    pr.run();
    pageRankData = pr.scores();
    pageRankData.resize(H.upperNodeIdBound(), 0.);
}

void CompositeCentrality::computeBetweenness(const Graph &H) {
    betweennessData.assign(H.upperNodeIdBound(), 0.);
    const count n = H.numberOfNodes();
    if (wBetweenness == 0. || n < 3)
        return;
    if (H.isDirected() && betweennessMethod != BetweennessMethod::EXACT)
        // Both sampling classes estimate the diameter, which NetworKit cannot do on directed
        // graphs yet.
        throw std::invalid_argument("approximate betweenness needs an undirected graph");
    // The sampling methods estimate the share of ordered node pairs, b / (n (n - 1)); the exact
    // normalised score is b / ((n - 1)(n - 2)) on ordered pairs, so the factor is n / (n - 2).
    const double toExact = static_cast<double>(n) / static_cast<double>(n - 2);
    switch (betweennessMethod) {
    case BetweennessMethod::EXACT: {
        Betweenness bc(H, true);
        bc.run();
        betweennessData = bc.scores();
        break;
    }
    case BetweennessMethod::APPROXIMATE: {
        ApproxBetweenness ab(H, epsilon, delta);
        ab.run();
        betweennessData = ab.scores();
        for (double &x : betweennessData)
            x *= toExact;
        break;
    }
    case BetweennessMethod::KADABRA: {
        KadabraBetweenness kb(H, epsilon, delta);
        kb.run();
        betweennessData = kb.scores();
        // KadabraBetweenness doubles its estimates on undirected graphs.
        for (double &x : betweennessData)
            x *= toExact / 2.;
        break;
    }
    }
    betweennessData.resize(H.upperNodeIdBound(), 0.);
}

void CompositeCentrality::computeDegree(const Graph &H) {
    degreeData.assign(H.upperNodeIdBound(), 0.);
    const count n = H.numberOfNodes();
    if (wDegree == 0. || n == 0)
        return;
    if (n == 1) {
        H.forNodes([&](node u) { degreeData[u] = 1.; });
        return;
    }
    const double scale = 1. / static_cast<double>(n - 1);
    H.parallelForNodes([&](node u) {
        const count d = H.isDirected() ? H.degreeOut(u) + H.degreeIn(u) : H.degree(u);
        degreeData[u] = static_cast<double>(d) * scale;
    });
}

void CompositeCentrality::computeCloseness(const Graph &H) {
    closenessData.assign(H.upperNodeIdBound(), 0.);
    if (wCloseness == 0. || H.numberOfNodes() < 2)
        return;
    if (closenessMethod == ClosenessMethod::EXACT) {
        // NetworkX uses inbound distances on directed graphs; NetworKit's Closeness uses
        // outbound distances, so it runs on the transpose.
        std::unique_ptr<Graph> transposed;
        if (H.isDirected())
            transposed = std::make_unique<Graph>(GraphTools::transpose(H));
        const Graph &T = H.isDirected() ? *transposed : H;
        Closeness cl(T, true, ClosenessVariant::GENERALIZED);
        cl.run();
        closenessData = cl.scores();
    } else {
        bool connected;
        if (H.isDirected()) {
            StronglyConnectedComponents scc(H);
            scc.run();
            connected = scc.numberOfComponents() == 1;
        } else {
            ConnectedComponents cc(H);
            cc.run();
            connected = cc.numberOfComponents() == 1;
        }
        if (!connected)
            throw std::runtime_error("approximate closeness needs a connected graph");
        ApproxCloseness ac(H, std::min(closenessSamples, H.numberOfNodes()), 0.1, true,
                           ApproxCloseness::ClosenessType::INBOUND);
        ac.run();
        closenessData = ac.scores();
    }
    closenessData.resize(H.upperNodeIdBound(), 0.);
}

void CompositeCentrality::run() {
    const count z = G.upperNodeIdBound();
    if (!histories.empty() && histories.size() != z)
        throw std::runtime_error("histories no longer match the graph");
    if (!attributes.empty() && attributes.size() != z)
        throw std::runtime_error("attributes no longer match the graph");

    // Edge weights are ignored: every component is computed on the unweighted graph.
    std::unique_ptr<Graph> unweightedCopy;
    if (G.isWeighted())
        unweightedCopy = std::make_unique<Graph>(G, false, G.isDirected());
    const Graph &H = unweightedCopy ? *unweightedCopy : G;
    computePageRank(H);
    computeBetweenness(H);
    computeDegree(H);
    computeCloseness(H);
    unweightedCopy.reset();

    if (scaleComponents) {
        divideByMax(G, pageRankData);
        divideByMax(G, betweennessData);
        divideByMax(G, degreeData);
        divideByMax(G, closenessData);
    }

    centralityData.assign(z, 0.);
    G.parallelForNodes([&](node u) {
        centralityData[u] = wPageRank * pageRankData[u] + wBetweenness * betweennessData[u]
                            + wDegree * degreeData[u] + wCloseness * closenessData[u];
    });
    divideByMax(G, centralityData);

    entropyData.assign(z, 0.);
    if (!histories.empty())
        G.parallelForNodes([&](node u) { entropyData[u] = demandEntropy(histories[u], bins); });

    attributeData.assign(z, 0.);
    if (!attributes.empty())
        G.forNodes([&](node u) { attributeData[u] = attributes[u]; });

    scoreData.assign(z, 0.);
    G.parallelForNodes([&](node u) {
        scoreData[u] =
            alpha * centralityData[u] + beta * entropyData[u] + gamma * attributeData[u];
    });

    hasRun = true;
}

double CompositeCentrality::maximum() {
    return alpha + beta + gamma;
}

const std::vector<double> &CompositeCentrality::centralityScores() const {
    assureFinished();
    return centralityData;
}

const std::vector<double> &CompositeCentrality::entropyScores() const {
    assureFinished();
    return entropyData;
}

const std::vector<double> &CompositeCentrality::attributeScores() const {
    assureFinished();
    return attributeData;
}

const std::vector<double> &CompositeCentrality::componentScores(Component component) const {
    assureFinished();
    switch (component) {
    case Component::PAGERANK:
        return pageRankData;
    case Component::BETWEENNESS:
        return betweennessData;
    case Component::DEGREE:
        return degreeData;
    case Component::CLOSENESS:
        return closenessData;
    }
    throw std::invalid_argument("unknown component");
}

} // namespace NetworKit
