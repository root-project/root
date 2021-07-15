const { Octokit } = require("octokit");
const { graphql } = require("@octokit/graphql");

const token = process.env.GITHUB_TOKEN;
const [owner, repo] = process.env.GITHUB_REPOSITORY.split("/");

const octokit = new Octokit({ auth: token });

async function nudge(issue_number) {
  const { data: issue_details } = await octokit.rest.issues.get({
    owner: owner,
    repo: repo,
    issue_number: issue_number,
  });
  const closed_by = issue_details.closed_by.login;
  const assignees = issue_details.assignees.map(user => user.login);

  // Put the person who closed the issue first, then all assignees (excluding
  // the person who closed, for obvious reasons).
  const ping_logins = [ closed_by ].concat(assignees.filter(user => user != closed_by));
  const ping = ping_logins.map(login => "@" + login).join(", ");

  console.log("Posting comment to issue #" + issue_number + "; pinging [ " + ping + " ]");

  let body = "Hi " + ping + ",\n\n";
  body += "It appears this issue is closed, but wasn't yet added to a project. ";
  body += "Please add upcoming versions that will include the fix, or 'not applicable' otherwise.";
  body += "\n\nSincerely,\n:robot:\n";

  octokit.rest.issues.createComment({
    owner: owner,
    repo: repo,
    issue_number: issue_number,
    body: body,
  });
}

async function iterate_issues() {
  const { search: { nodes: issues } } = await graphql(
    `
      {
        search(query: "repo:root-project/root is:issue is:closed no:project", type: ISSUE, first: 100) {
          nodes {
            ... on Issue {
              number
            }
          }
        }
      }
    `,
    {
      headers: {
        authorization: "token " + token,
      },
    }
  );

  for (const issue of issues) {
    nudge(issue.number);
  }
}

iterate_issues();
