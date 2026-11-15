// Rhizome's signed approvals (rhizome_trust.h): endorsements, pinning, delegation to successors, revocation. Prints a line a
// check and a JSON summary; exits non-zero on a failure.
#include "rhizome_trust.h"
#include <cstdio>

using namespace xpu2::interactive::rhizome;

static int failures = 0, checks = 0;
static void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / "rhizome-trust-test";
    std::filesystem::remove_all(root);
    auto master = Identity::make(), node = Identity::make(), other = Identity::make(), successor = Identity::make(), rogue = Identity::make();

    Trust m(root / "master", *master);
    m.require(true);
    m.trustSelf();
    check(m.required() && m.approver(), "a master with --approve requires endorsements and may give them");
    check(!m.admits(node->hash(), {}), "a node without an endorsement is held");

    const Bytes e = m.endorse(node->hash());
    check(m.admits(node->hash(), {e}), "the master's endorsement admits the node");
    check(!m.admits(other->hash(), {e}), "another hash showing the node's endorsement is not admitted");
    Bytes forged = e;
    forged[10] ^= 1;
    check(!m.admits(node->hash(), {forged}), "a changed endorsement does not verify");
    check(m.issued(node->hash()).has_value(), "the endorsement is kept in the master's folder");

    // The node: it trusts no approver until its first endorsement, from the master it joined.
    Trust n(root / "node", *node);
    check(!n.required(), "a node trusting no approver asks nothing of those it meets");
    check(!n.take(e, rogue->hash()), "an endorsement passed on by another member is not pinned");
    check(n.take(e, master->hash()), "the master's endorsement is taken from the master");
    check(n.trusted(master->hash()) && n.required(), "the master is pinned as its approver, and endorsements are now asked");
    Trust again(root / "node", *node);
    check(again.trusted(master->hash()) && again.shown().size() == 1, "the node keeps its approver and endorsement across restarts");

    // A rogue approver's endorsement is not trusted by members of this rhizome.
    Trust r(root / "rogue", *rogue);
    r.trustSelf();
    const Bytes re = r.endorse(other->hash());
    check(!m.admits(other->hash(), {re}) && !n.admits(other->hash(), {re}), "an endorsement from an approver no one trusts admits nothing");

    // A successor: endorsed as an approver, it endorses in its turn; members follow the delegation.
    const Bytes se = m.endorse(successor->hash(), Role::Approver);
    Trust s(root / "successor", *successor);
    check(s.take(se, master->hash()) && s.approver(), "a successor endorsed as an approver may approve");
    const Bytes oe = s.endorse(other->hash());
    check(n.admits(other->hash(), {oe, se}), "a member admits a node the successor endorsed, by its delegation");
    check(!n.admits(other->hash(), {oe}), "but not without the successor's own endorsement to follow");

    // Revocation: signed by a trusted approver, it beats the endorsements issued before it, wherever it is taken.
    const Bytes rv = m.revoke(node->hash());
    check(!m.admits(node->hash(), {e}), "a revoked node is not admitted by the master");
    check(!m.issued(node->hash()).has_value(), "the endorsement the master issued it no longer counts");
    auto subject = s.takeRevocation(rv);
    check(subject && *subject == node->hash() && !s.admits(node->hash(), {e}), "the revocation passed on is kept and enforced by a member");
    const Bytes fake = r.revoke(successor->hash());
    check(!n.takeRevocation(fake), "a revocation from an untrusted approver is refused");
    std::filesystem::remove_all(root);
    std::printf("{\"checks\": %d, \"failures\": %d}\n", checks, failures);
    return failures ? 1 : 0;
}
