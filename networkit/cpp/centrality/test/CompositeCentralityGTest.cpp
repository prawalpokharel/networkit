/*
 * CompositeCentralityGTest.cpp
 *
 *  Created on: 15.11.2026
 *      Author: Prawal Pokharel
 */

// networkit-format

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <networkit/auxiliary/Random.hpp>
#include <networkit/centrality/Betweenness.hpp>
#include <networkit/centrality/Closeness.hpp>
#include <networkit/centrality/CompositeCentrality.hpp>
#include <networkit/centrality/PageRank.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/generators/ErdosRenyiGenerator.hpp>
#include <networkit/graph/GraphTools.hpp>

namespace NetworKit {

class CompositeCentralityGTest : public testing::Test {};

namespace {

using Component = CompositeCentrality::Component;

Graph pathGraph(count n) {
    Graph G(n);
    for (node u = 0; u + 1 < n; ++u)
        G.addEdge(u, u + 1);
    return G;
}

Graph largestComponent(const Graph &G) {
    return ConnectedComponents::extractLargestConnectedComponent(G, true);
}

} // namespace

TEST_F(CompositeCentralityGTest, testHandComputedPath) {
    // Path 0 - 1 - 2 with the NetworkX normalisations, solved by hand.
    const Graph G = pathGraph(3);
    CompositeCentrality cc(G);
    cc.run();

    const double pr1 = 0.135 / 0.2775; // x1 = 0.05 + 0.85 * 2 * x0, x0 = 0.05 + 0.425 * x1
    const double pr0 = 0.05 + 0.425 * pr1;
    const double end = 0.35 * pr0 + 0.20 * 0.5 + 0.15 * (2. / 3.);
    const double mid = 0.35 * pr1 + 0.30 * 1. + 0.20 * 1. + 0.15 * 1.;

    const auto &pageRank = cc.componentScores(Component::PAGERANK);
    EXPECT_NEAR(pageRank[0], pr0, 1e-8);
    EXPECT_NEAR(pageRank[1], pr1, 1e-8);
    EXPECT_NEAR(cc.componentScores(Component::BETWEENNESS)[1], 1., 1e-12);
    EXPECT_NEAR(cc.componentScores(Component::DEGREE)[0], 0.5, 1e-12);
    EXPECT_NEAR(cc.componentScores(Component::CLOSENESS)[0], 2. / 3., 1e-12);

    const auto &c = cc.centralityScores();
    EXPECT_NEAR(c[1], 1., 1e-12);
    EXPECT_NEAR(c[0], end / mid, 1e-8);
    EXPECT_NEAR(c[2], end / mid, 1e-8);
    EXPECT_NEAR(cc.score(1), 0.40, 1e-12);
    EXPECT_NEAR(cc.score(0), 0.40 * end / mid, 1e-8);
    EXPECT_DOUBLE_EQ(cc.maximum(), 1.);
}

TEST_F(CompositeCentralityGTest, testMatchesComponentAlgorithms) {
    Aux::Random::setSeed(42, false);
    for (bool directed : {false, true}) {
        // Undirected: no isolated nodes, so plain PageRank equals the sink-handling variant.
        const Graph G = directed ? ErdosRenyiGenerator(120, 0.04, true).generate()
                                 : largestComponent(ErdosRenyiGenerator(120, 0.04, false).generate());
        for (bool scale : {false, true}) {
            CompositeCentrality cc(G, 0.40, 0.35, 0.25, scale);
            cc.run();

            Betweenness bc(G, true);
            bc.run();
            PageRank pr(G, 0.85, 1e-9, false, PageRank::SinkHandling::DISTRIBUTE_SINKS);
            pr.run();
            const Graph T = directed ? GraphTools::transpose(G) : G;
            Closeness cl(T, true, ClosenessVariant::GENERALIZED);
            cl.run();

            std::vector<double> b = bc.scores(), p = pr.scores(), c = cl.scores(), d(G.numberOfNodes());
            G.forNodes([&](node u) {
                d[u] = static_cast<double>(directed ? G.degreeIn(u) + G.degreeOut(u) : G.degree(u))
                       / static_cast<double>(G.numberOfNodes() - 1);
            });
            if (scale) {
                for (auto *v : {&b, &p, &c, &d}) {
                    const double top = *std::max_element(v->begin(), v->end());
                    for (double &x : *v)
                        x /= top;
                }
            }
            std::vector<double> composite(G.numberOfNodes());
            G.forNodes([&](node u) {
                composite[u] = 0.35 * p[u] + 0.30 * b[u] + 0.20 * d[u] + 0.15 * c[u];
            });
            const double top = *std::max_element(composite.begin(), composite.end());
            G.forNodes([&](node u) {
                EXPECT_NEAR(cc.centralityScores()[u], composite[u] / top, 1e-9);
                EXPECT_NEAR(cc.score(u), 0.40 * composite[u] / top, 1e-9);
            });
        }
    }
}

TEST_F(CompositeCentralityGTest, testDemandEntropy) {
    using CC = CompositeCentrality;
    EXPECT_DOUBLE_EQ(CC::demandEntropy({}), 0.);
    EXPECT_DOUBLE_EQ(CC::demandEntropy({0.42, 0.42, 0.42}), 0.);
    // One value in each of the ten bins gives the maximum.
    EXPECT_NEAR(CC::demandEntropy({0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95}),
                1., 1e-12);
    // Two equally filled bins.
    EXPECT_NEAR(CC::demandEntropy({0.05, 0.95}), 1. / std::log2(10.), 1e-12);
    // 1.0 falls in the last bin, values outside [0, 1] are clipped.
    EXPECT_DOUBLE_EQ(CC::demandEntropy({1.0, 0.95}), 0.);
    EXPECT_DOUBLE_EQ(CC::demandEntropy({-1., 0.}), 0.);
    EXPECT_DOUBLE_EQ(CC::demandEntropy({2., 1.}), 0.);
    // A value on a bin edge goes to the bin it opens.
    EXPECT_DOUBLE_EQ(CC::demandEntropy({0.5, 0.55}), 0.);
    EXPECT_NEAR(CC::demandEntropy({0.5, 0.45}), 1. / std::log2(10.), 1e-12);
    // Fewer bins.
    EXPECT_NEAR(CC::demandEntropy({0.1, 0.9}, 2), 1., 1e-12);
    EXPECT_DOUBLE_EQ(CC::demandEntropy({0.1, 0.9}, 1), 0.);
}

TEST_F(CompositeCentralityGTest, testHistoriesAndAttributes) {
    const Graph G = pathGraph(3);
    std::vector<std::vector<double>> histories = {{0.05, 0.95}, {}, {0.3, 0.3}};
    std::vector<double> attributes = {1., 0.5, 0.};
    CompositeCentrality cc(G, 0.40, 0.35, 0.25);
    cc.setDemandHistories(histories);
    cc.setNodeAttributes(attributes);
    cc.run();

    const auto &c = cc.centralityScores();
    const auto &h = cc.entropyScores();
    EXPECT_NEAR(h[0], 1. / std::log2(10.), 1e-12);
    EXPECT_DOUBLE_EQ(h[1], 0.);
    EXPECT_DOUBLE_EQ(h[2], 0.);
    EXPECT_DOUBLE_EQ(cc.attributeScores()[1], 0.5);
    G.forNodes([&](node u) {
        EXPECT_NEAR(cc.score(u), 0.40 * c[u] + 0.35 * h[u] + 0.25 * attributes[u], 1e-12);
    });
}

TEST_F(CompositeCentralityGTest, testInvalidArguments) {
    const Graph G = pathGraph(4);
    EXPECT_THROW(CompositeCentrality(G, -0.1), std::invalid_argument);
    CompositeCentrality cc(G);
    EXPECT_THROW(cc.setComponentWeights(0., 0., 0., 0.), std::invalid_argument);
    EXPECT_THROW(cc.setComponentWeights(-1., 1., 1., 1.), std::invalid_argument);
    EXPECT_THROW(cc.setBetweennessMethod(CompositeCentrality::BetweennessMethod::KADABRA, 0.),
                 std::invalid_argument);
    EXPECT_THROW(cc.setClosenessMethod(CompositeCentrality::ClosenessMethod::APPROXIMATE, 0),
                 std::invalid_argument);
    EXPECT_THROW(cc.setDemandHistories({{0.1}}), std::invalid_argument);
    EXPECT_THROW(cc.setNodeAttributes({1.}), std::invalid_argument);
    EXPECT_THROW(cc.centralityScores(), std::runtime_error);
}

TEST_F(CompositeCentralityGTest, testComponentWeights) {
    // With only the degree component, C is the degree divided by its maximum.
    Aux::Random::setSeed(7, false);
    const Graph G = ErdosRenyiGenerator(80, 0.06, false).generate();
    CompositeCentrality cc(G);
    cc.setComponentWeights(0., 0., 1., 0.);
    cc.run();
    count maxDegree = 0;
    G.forNodes([&](node u) { maxDegree = std::max(maxDegree, G.degree(u)); });
    G.forNodes([&](node u) {
        EXPECT_NEAR(cc.centralityScores()[u],
                    static_cast<double>(G.degree(u)) / static_cast<double>(maxDegree), 1e-12);
        EXPECT_DOUBLE_EQ(cc.componentScores(Component::BETWEENNESS)[u], 0.);
    });
}

TEST_F(CompositeCentralityGTest, testApproximateMethods) {
    Aux::Random::setSeed(3, false);
    const Graph G = largestComponent(ErdosRenyiGenerator(600, 0.01, false).generate());

    CompositeCentrality exact(G);
    exact.run();
    const auto &bExact = exact.componentScores(Component::BETWEENNESS);
    const auto &cExact = exact.componentScores(Component::CLOSENESS);
    double sumExact = 0.;
    G.forNodes([&](node u) { sumExact += bExact[u]; });

    for (auto method : {CompositeCentrality::BetweennessMethod::APPROXIMATE,
                        CompositeCentrality::BetweennessMethod::KADABRA}) {
        SCOPED_TRACE(method == CompositeCentrality::BetweennessMethod::KADABRA ? "kadabra"
                                                                              : "approximate");
        CompositeCentrality approx(G);
        approx.setBetweennessMethod(method, 0.005, 0.1);
        approx.setClosenessMethod(CompositeCentrality::ClosenessMethod::APPROXIMATE, 200);
        approx.run();
        const auto &b = approx.componentScores(Component::BETWEENNESS);
        const auto &c = approx.componentScores(Component::CLOSENESS);
        double sum = 0., maxDiffB = 0., maxDiffC = 0., maxDiffScore = 0.;
        G.forNodes([&](node u) {
            sum += b[u];
            maxDiffB = std::max(maxDiffB, std::abs(b[u] - bExact[u]));
            maxDiffC = std::max(maxDiffC, std::abs(c[u] - cExact[u]));
            maxDiffScore = std::max(maxDiffScore, std::abs(approx.score(u) - exact.score(u)));
        });
        // Rescaled estimates must sum to the exact total (the scale factor is right).
        EXPECT_NEAR(sum / sumExact, 1., 0.05);
        EXPECT_LT(maxDiffB, 0.02);
        EXPECT_LT(maxDiffC, 0.05);
        EXPECT_LT(maxDiffScore, 0.05);
    }
}

TEST_F(CompositeCentralityGTest, testApproximateBetweennessNeedsUndirectedGraph) {
    Aux::Random::setSeed(5, false);
    const Graph G = ErdosRenyiGenerator(100, 0.05, true).generate();
    for (auto method : {CompositeCentrality::BetweennessMethod::APPROXIMATE,
                        CompositeCentrality::BetweennessMethod::KADABRA}) {
        CompositeCentrality cc(G);
        cc.setBetweennessMethod(method);
        EXPECT_THROW(cc.run(), std::invalid_argument);
    }
}

TEST_F(CompositeCentralityGTest, testApproximateClosenessNeedsConnectedGraph) {
    Graph G = pathGraph(4);
    G.addNode();
    CompositeCentrality cc(G);
    cc.setClosenessMethod(CompositeCentrality::ClosenessMethod::APPROXIMATE);
    EXPECT_THROW(cc.run(), std::runtime_error);
}

TEST_F(CompositeCentralityGTest, testTinyAndEmptyGraphs) {
    Graph empty(0);
    CompositeCentrality ce(empty);
    ce.run();
    EXPECT_TRUE(ce.scores().empty());

    Graph single(1);
    CompositeCentrality cs(single);
    cs.run();
    EXPECT_DOUBLE_EQ(cs.centralityScores()[0], 1.);
    EXPECT_DOUBLE_EQ(cs.score(0), 0.40);

    Graph pair(2);
    pair.addEdge(0, 1);
    CompositeCentrality cp(pair);
    cp.run();
    EXPECT_DOUBLE_EQ(cp.centralityScores()[0], 1.);
    EXPECT_DOUBLE_EQ(cp.centralityScores()[1], 1.);
}

TEST_F(CompositeCentralityGTest, testIgnoresEdgeWeights) {
    Graph weighted(5, true);
    Graph unweighted(5);
    const std::vector<std::pair<node, node>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {1, 3}};
    for (auto [u, v] : edges) {
        weighted.addEdge(u, v, 5. + static_cast<double>(u));
        unweighted.addEdge(u, v);
    }
    CompositeCentrality cw(weighted), cu(unweighted);
    cw.run();
    cu.run();
    for (node u = 0; u < 5; ++u)
        EXPECT_NEAR(cw.score(u), cu.score(u), 1e-12);
}

TEST_F(CompositeCentralityGTest, testDeletedNodes) {
    Graph G = pathGraph(5);
    G.removeNode(4);
    CompositeCentrality cc(G);
    cc.run();
    EXPECT_EQ(cc.scores().size(), G.upperNodeIdBound());
    EXPECT_DOUBLE_EQ(cc.scores()[4], 0.);

    // Same scores as the path on the remaining four nodes.
    const Graph P = pathGraph(4);
    CompositeCentrality ref(P);
    ref.run();
    for (node u = 0; u < 4; ++u)
        EXPECT_NEAR(cc.score(u), ref.score(u), 1e-9);
}

TEST_F(CompositeCentralityGTest, testUndirectedIsolatedNodeIsSink) {
    // NetworkX spreads the PageRank of an isolated node over all nodes, so the scores sum to 1.
    Graph G = pathGraph(3);
    G.addNode();
    CompositeCentrality cc(G);
    cc.run();
    const auto &pr = cc.componentScores(Component::PAGERANK);
    double sum = 0.;
    G.forNodes([&](node u) { sum += pr[u]; });
    EXPECT_NEAR(sum, 1., 1e-8);
    EXPECT_GT(pr[3], 0.);
}

} // namespace NetworKit
